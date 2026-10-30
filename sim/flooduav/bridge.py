"""FloodUAV bridge: runs the patched Unity build and serves it to the mission server.

The mission server (sar/mission/flooduav.py) talks to this bridge the way it talks to the
Isaac bridge: sim time and video streams, plus a drone state and a
setpoint endpoint for the two Unity drones. See sim/flooduav/README.md.

  GET  /time               {"sim_time": s}
  GET  /info               scene info: boundary, start, survivors, area polygon, map image box, cell types
                           (mission meters; cells: sample (i, j) sits at origin_en + i di_en + j dj_en)
  GET  /state              drone poses in mission meters, episode, found survivors
  POST /setpoint           {"drone": k, "pos": [e, n, u], "vel": [ve, vn, vu]}; "hold": true holds where it is,
                           "land": true descends and rests on the ground below
  POST /speed              {"speed": m/s} horizontal speed cap for both drones
  GET  /agents             {"onnx": any agent, "agent": id, "agents": [{"id", "label"}]}: the trained agents now
                           in --models (drop-in, rescanned on each call)
  POST /pilot              {"pilot": "onnx" | "setpoint", "limits": "training" | "bridge", "agent": id} who flies
                           the search drone: a trained agent (--models, default --agent) on Unity's observation, or the setpoints;
                           a setpoint for drone 0 hands it back to the setpoints. "training": the build's flight
                           limits while the agent flies; "bridge": --max-speed (or the /speed cap), --climb, --tilt
  POST /reset              restart Unity's mission: drones back on their pads, survivors unfound
  POST /survivors          {"seed": n, "count": k, "shares": {...}} new survivor layout of k survivors, then a
                           reset; no seed: a random one, -1: the build's fixed survivors; no count: the build's 15.
                           shares: terrain mix weights by SHARE_KEYS, normalized (none: the default mix).
                           Returns once the new positions are in /info.
  POST /camera             {"view": "follow", "drone": k, "distance", "yaw", "pitch"} or {"view": "drone", "drone": k, "fov", "pan", "tilt"}
  GET  /stream/follow      main Unity view, following drone 0
  GET  /stream/follow1     follow view of drone 1
  GET  /stream/cam0, cam1  drone cameras (0 search drone, 1 the build's pickup drone)
  GET  /stream/overview    square top-down view
                           ?small=1: at most SMALL_W wide and SMALL_FPS frames a second (the UI's fallback)
  GET  /h264/<region>      the same views as H.264 (follow, follow1, cam0, cam1, overview): frames of
                           [4-byte big-endian length][1 byte: 1 key frame, 0 not][Annex B access unit]
  GET  /overview.jpg       one top-down frame from the live stream
  GET  /map.jpg            one high resolution top-down render (--map-px), for the map background

Mission frame: east = boundary +x, north = boundary +z, up = boundary +y, in meters, with the
origin at the drone start point. Run with .venv-fly and DISPLAY set (sim/flooduav/run.sh).
"""

import argparse
import json
import math
import os
import queue
import re
import struct
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.side_channel import IncomingMessage, OutgoingMessage, SideChannel

import agents

# observation indices (SurvivorSearchAgent.CollectObservations)
VEL = slice(0, 3)          # local velocity / (10, 3, 10)
AGL_HIT, AGL = 9, 10       # down ray hit flag, distance / 100 m
POS = slice(11, 14)        # position in the boundary, normalized to [-1, 1]
SIN, COS = 14, 15          # heading from boundary forward (north), clockwise
RAYS = slice(16, 24)       # 8 horizontal obstacle rays / 30 m, from forward clockwise
UP_RAY, DOWN_RAY = 24, 25
SIM_DT = 0.1               # one decision = 5 physics steps of 0.02 s
FENCE_S = 3.0              # the agent is steered back inside when it would leave the boundary within this time
STUCK_S = 4.0              # the agent's drone is lifted when it moves under 0.5 m in this time (resting on a pipe or roof)
ROOT = Path(__file__).resolve().parents[2]

# screen layout from Hook.SplitViews: follow view of drone 0, the two drone cameras stacked,
# square overview, follow view of drone 1
W, H = 3920, 720
REGIONS = {"follow": (0, 0, 1280, 720), "cam0": (1280, 0, 640, 360), "cam1": (1280, 360, 640, 360),
           "overview": (1920, 0, 720, 720), "follow1": (2640, 0, 1280, 720)}
SMALL_W, SMALL_FPS = 640, 6   # /stream/<region>?small=1
AUD = b"\x00\x00\x01\x09"      # access unit delimiter: each H.264 frame starts with one
NAL = re.compile(rb"\x00\x00\x01(.)", re.S)
RECORDED = re.compile(r"Recorded (\S+) at")
FINISHED = re.compile(r"Mission (Failed|TimedOut|Succeeded): (.*)")
# survivor terrain mix keys, as in sar/prior/flood_bank.py SHARES and the patch's Layout.ShareKeys
SHARE_KEYS = ("roofs", "high_ground", "bridges", "water", "uniform")


def parse_shares(shares):
    """Terrain mix weights {key: w} as a normalized list in SHARE_KEYS order; None stays None."""
    if shares is None:
        return None
    if not isinstance(shares, dict) or set(shares) - set(SHARE_KEYS):
        raise ValueError(f"shares: an object with keys from {', '.join(SHARE_KEYS)}")
    w = [float(shares.get(k, 0.0)) for k in SHARE_KEYS]
    if any(not math.isfinite(x) or x < 0 for x in w) or sum(w) <= 0:
        raise ValueError("shares: weights must be 0 or more, and not all 0")
    return [x / sum(w) for x in w]


class Scene:
    """Boundary frame from the info file the patch writes at each mission start."""

    def __init__(self, info: dict, margin: float):
        b = info["boundary"]
        self.scale = np.array(b["scale"])
        self.center, self.half = np.array(b["center"]), np.array(b["size"]) / 2
        self.start = np.array(info["start"])
        self.info = info
        lo = (self.center - self.half * (1 - margin) - self.start) * self.scale
        hi = (self.center + self.half * (1 - margin) - self.start) * self.scale
        self.lo, self.hi = lo[[0, 2, 1]], hi[[0, 2, 1]]       # (e, n, u) limits

    def to_mission(self, local) -> np.ndarray:
        d = (np.asarray(local, float) - self.start) * self.scale
        return d[[0, 2, 1]]

    def from_norm(self, norm) -> np.ndarray:
        return self.to_mission(self.center + np.asarray(norm, float) * self.half)

    def survivors(self):
        return {s["id"]: self.to_mission(s["local"]).tolist() for s in self.info["survivors"] if s["searchable"]}

    def cells(self):
        """Cell type samples of the patch (Layout.cs) on the mission frame, or None."""
        c = self.info.get("cells")
        if not c:
            return None
        o = np.array(c["origin"], float)
        o0 = self.to_mission(o)
        di = self.to_mission(o + [c["dx"], 0, 0]) - o0
        dj = self.to_mission(o + [0, 0, c["dz"]]) - o0
        return {"origin_en": o0[:2].tolist(), "di_en": di[:2].tolist(), "dj_en": dj[:2].tolist(),
                "nx": c["nx"], "nz": c["nz"], "types": c["types"]}

    def area(self):
        lo, hi = self.lo, self.hi
        return [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]]]

    def overview_box(self):
        """Ground box (e, n) of the overview camera: orthographic, square, centered on the boundary."""
        ext = self.half * self.scale
        hh = hw = max(ext[2], ext[0])
        c = self.to_mission(self.center)
        return [[c[0] - hw, c[1] + hh], [c[0] + hw, c[1] + hh], [c[0] + hw, c[1] - hh], [c[0] - hw, c[1] - hh]]


class FleetChannel(SideChannel):
    """Pickup drone commands and state (sim/flooduav/patch/FloodPatch/Fleet.cs)."""

    def __init__(self):
        super().__init__(uuid.UUID("6f1c2b6e-5a0d-4e8b-9a43-0d2f4c7b9e11"))
        self.latest = None

    def on_message_received(self, msg: IncomingMessage):
        self.latest = msg.read_float32_list()

    def send(self, values):
        m = OutgoingMessage()
        m.write_float32_list([float(v) for v in values])
        self.queue_message_to_send(m)


class Drone:
    """One drone in mission meters. mode: land (descend and rest on the ground), hold, fly."""

    def __init__(self, home):
        self.home = np.asarray(home, float).copy()
        self.pos, self.vel = self.home.copy(), np.zeros(3)
        self.sin, self.cos = 0.0, 1.0
        self.agl = None              # m above whatever is below, None: nothing within range
        self.ray = 30.0              # nearest horizontal obstacle, m
        self.mode, self.target, self.hold_at = "land", None, None
        self.cmd = [0.0, 0.0, 0.0, 0.0]  # last normalized command, for /state

    @property
    def landed(self):
        # pushed down and not moving: resting on something (the pickup pad sits above the ray's hit)
        return bool(self.mode == "land" and self.agl is not None and self.agl < 5.0 and np.abs(self.vel).max() < 0.3)

    def snapshot(self):
        return {"pos": self.pos.round(2).tolist(), "vel": self.vel.round(2).tolist(),
                "yaw": round(math.degrees(math.atan2(self.sin, self.cos)) % 360, 1),
                "agl": None if self.agl is None else round(self.agl, 1), "mode": self.mode, "landed": self.landed,
                "target": None if self.target is None else self.target[0].round(1).tolist(),
                "ray": round(self.ray, 1), "cmd": [round(c, 2) for c in self.cmd]}


class Unity:
    """Owns the Unity process. One thread steps it at `rate` x real time.

    Drone 0 is the search drone (ML-Agents actions), drone 1 the build's pickup drone (side channel).
    """

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.steps, self.episode, self.reason = 0, 0, ""
        self.found: dict[str, list] = {}
        self.speed_cap = args.max_speed
        self.pending_reset = False
        self.pending_layout = None
        self.layout_done = threading.Event()
        self.layout_sent = False
        self.pilot, self.limits = "setpoint", "training"   # search drone: who flies it, which flight limits
        # trained search agents dropped into --models, the chosen one and its loaded model
        self.catalog = agents.Catalog(args.models)
        self.agent, self.flying = args.agent, None
        self._still, self._climb_until = (0.0, np.zeros(3)), -1.0   # stuck check while the agent flies
        self._limits_sent = None
        os.environ.update({
            "FLOODUAV_HSPEED": str(args.max_speed), "FLOODUAV_VSPEED": str(args.climb),
            "FLOODUAV_TILT": str(args.tilt), "FLOODUAV_TIME_LIMIT": "1000000",
            "FLOODUAV_SPLIT": "1", "FLOODUAV_INFO": args.info, "FLOODUAV_FLEET": "1", "FLOODUAV_SOFT": "1",
            "FLOODUAV_FOLLOW": args.follow, "FLOODUAV_AIRCRAFT": f"wingArea={args.wing_area}",
            "FLOODUAV_MAP": args.map_file, "FLOODUAV_SURVIVOR_SEED": str(args.survivor_seed),
            "FLOODUAV_SURVIVOR_COUNT": str(args.survivor_count),
        })
        Path(args.info).unlink(missing_ok=True)
        ch = EngineConfigurationChannel()
        self.fleet = FleetChannel()
        self.env = UnityEnvironment(
            file_name=args.build, no_graphics=False, worker_id=args.worker, timeout_wait=180,
            side_channels=[ch, self.fleet],
            additional_args=["-screen-fullscreen", "0", "-screen-width", str(W), "-screen-height", str(H),
                             "-logFile", args.unity_log, *args.unity_args.split()])
        ch.set_configuration_parameters(width=W, height=H, quality_level=args.quality,
                                        time_scale=args.time_scale, target_frame_rate=30)
        self.env.reset()
        # The search drone; the pickup drone has no remote behavior (Fleet.cs flies it).
        names = list(self.env.behavior_specs)
        self.name = next((n for n in names if "pickup" not in n.lower()), names[0])
        print("behaviors:", names, "search:", self.name, flush=True)
        info = json.loads(Path(args.info).read_text())
        self.scene = Scene(info, args.margin)
        homes = [np.zeros(3)] + ([self.scene.to_mission(info["pickup"])] if "pickup" in info else [])
        self.drones = [Drone(h) for h in homes]
        self._log = open(args.unity_log, errors="replace")
        self.obs = None
        self.env.step()               # the first fleet state message comes with this step
        self._read_obs()
        self.home = self.drones[0].pos.copy()

    # ------------------------------------------------------------------ state
    def _read_obs(self):
        """Unity calls run outside the lock: HTTP threads only wait for the state update."""
        while True:
            dec, term = self.env.get_steps(self.name)
            ended = len(term) > 0
            o = (term if ended else dec).obs[0][0]
            f = self.fleet.latest
            with self.lock:
                self.obs = o
                d = self.drones[0]
                d.sin, d.cos = float(o[SIN]), float(o[COS])
                fwd, right = np.array([d.sin, d.cos]), np.array([d.cos, -d.sin])
                lv = o[VEL] * np.array([10.0, 3.0, 10.0])          # right, up, forward
                d.pos = self.scene.from_norm(o[POS])
                d.vel = np.array([*(right * lv[0] + fwd * lv[2]), lv[1]])
                d.agl = float(o[AGL]) * 100 if o[AGL_HIT] > 0.5 else None
                d.ray = float(min(o[RAYS].min(), o[DOWN_RAY])) * 30
                if f is not None and len(self.drones) > 1 and len(f) >= 11:
                    p = self.drones[1]
                    p.pos = self.scene.from_norm(f[0:3])
                    p.vel = np.array([f[3], f[5], f[4]], float)
                    p.sin, p.cos = float(f[6]), float(f[7])
                    p.agl = float(f[10]) * 150 if f[9] > 0.5 else None
                    p.ray = float(min(f[8] * 30, p.agl if p.agl is not None else 150))
                self._tail_log()
                if ended:
                    self.episode += 1
                    self.found = {}
                    self._land_all()
                    self._limits_sent = None
            if not ended:
                return
            self.env.step()          # ML-Agents starts the next episode; the drones are back on their pads

    def _land_all(self):
        for d in self.drones:
            d.mode, d.target, d.hold_at = "land", None, None
        self.pilot = "setpoint"

    def _tail_log(self):
        for line in self._log.readlines():
            if m := RECORDED.search(line):
                sid = m.group(1)
                pos = self.scene.survivors().get(sid)
                if pos is not None:
                    self.found[sid] = pos
            elif m := FINISHED.search(line):
                self.reason = f"{m.group(1)}: {m.group(2).strip()}"

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "t": round(self.steps * SIM_DT, 2), "episode": self.episode, "last_end": self.reason,
                "speed_cap": self.speed_cap, "pilot": self.pilot, "limits": self.limits, "agent": self.agent,
                "drones": [d.snapshot() for d in self.drones],
                "found": [{"id": k, "e": v[0], "n": v[1]} for k, v in self.found.items()],
            }

    # ---------------------------------------------------------------- control
    def set_target(self, k, pos=None, vel=None, mode="fly"):
        with self.lock:
            d = self.drones[k]
            if k == 0:
                self.pilot = "setpoint"
            d.mode, d.hold_at = mode, d.pos.copy()
            d.target = None if pos is None else (np.asarray(pos, float), np.asarray(vel if vel is not None else [0, 0, 0], float))

    def set_speed(self, v):
        with self.lock:
            self.speed_cap = float(np.clip(v, 1.0, self.args.max_speed))

    def agents(self) -> dict:
        """The trained agents now in --models (agents.Catalog); the default one first."""
        found = self.catalog.scan(len(self.obs)) if self.obs is not None else {}
        if self.agent not in found and found:
            self.agent = self.args.agent if self.args.agent in found else next(iter(found))
        return dict(sorted(found.items(), key=lambda kv: kv[0] != self.agent))

    def set_pilot(self, pilot, limits=None, agent=None):
        if pilot not in ("onnx", "setpoint") or limits not in (None, "training", "bridge"):
            raise ValueError("pilot: onnx or setpoint; limits: training or bridge")
        found = self.agents()
        agent = agent or self.agent
        if agent not in found:
            raise ValueError(f"no agent {agent!r} in {self.args.models}; agents: {', '.join(found) or 'none'}")
        with self.lock:
            self.agent, self.flying = agent, found[agent]["pilot"]
            if pilot == "onnx":
                d = self.drones[0]
                d.mode, d.target, d.hold_at = "fly", None, None
            self.pilot = pilot
            if limits is not None:
                self.limits = limits

    def _search_limits(self):
        """Fleet message 7 for the search drone: the build's limits (-1) under the agent, else the bridge's."""
        a = self.args
        if self.pilot == "onnx" and self.limits == "training":
            return (-1.0, -1.0, -1.0, -1.0)
        h = self.speed_cap if self.pilot == "onnx" else a.max_speed
        return (h, a.climb, a.tilt, a.wing_area)

    def reset(self):
        """Restart Unity's mission: drones back on their pads, survivors unfound."""
        with self.lock:
            self.pending_reset = True

    def new_layout(self, seed, count=-1, shares=None, timeout=60.0) -> bool:
        """Survivor layout `seed` with `count` survivors and terrain mix `shares` (list in SHARE_KEYS order,
        None: the default) from the next mission start on; waits until /info has it."""
        with self.lock:
            self.layout_done.clear()
            self.pending_layout = (int(seed), int(count), *(shares or []))
        return self.layout_done.wait(timeout)

    def _reload_info(self):
        """The patch rewrites the info file at each mission start, with the current survivor positions."""
        info = json.loads(Path(self.args.info).read_text())
        with self.lock:
            self.scene.info = info

    def _command(self, d: Drone):
        a, sc = self.args, self.scene
        landing = d.mode == "land"
        if d.target is None or landing:
            if d.hold_at is None:
                d.hold_at = d.pos.copy()
            tgt, ff = d.hold_at.copy(), np.zeros(3)
        else:
            tgt, ff = d.target
            tgt = tgt.copy()
        tgt = np.clip(tgt, sc.lo, sc.hi)
        tgt[2] = np.clip(tgt[2], a.min_alt, min(a.max_alt, sc.hi[2]))
        err = tgt - d.pos
        # Approach speed limited by the braking distance: the drone brakes at about
        # g*tan(tilt) at best, and a plain P gain overshoots and swings around the target.
        # Unity's velocity loop lags (about 0.4 s, then a 1.7 s time constant), so the
        # horizontal error is taken from where the drone will be `lead` seconds ahead.
        herr = err[:2] - d.vel[:2] * a.lead
        dist = float(np.hypot(*herr))
        cap = 3.0 if landing else self.speed_cap
        v = ff[:2] + (herr / dist * min(a.kp * dist, math.sqrt(2 * a.brake * dist)) if dist > 1e-3 else 0.0)
        sp = float(np.hypot(*v))
        if sp > cap:
            v *= cap / sp
        if landing:
            # straight down at the hold point; keep pushing down once on the ground so it stays put
            near = d.agl is not None and d.agl < 6.0
            up = -0.25 if near else -0.6
            if d.agl is not None and d.agl < 5.0:
                v = np.zeros(2)
            elif d.agl is None:          # nothing below within ray range (open water): hover, don't sink
                up = float(np.clip((max(d.hold_at[2], a.min_alt) - d.pos[2]) / a.climb, -1, 1))
        else:
            up = float(np.clip((ff[2] + 1.0 * err[2]) / a.climb, -1, 1))
            if d.ray < 12.0:                 # something within 12 m: climb and slow down
                up, v = 1.0, v * 0.3
        fwd = (v[0] * d.sin + v[1] * d.cos) / a.max_speed
        side = (v[0] * d.cos - v[1] * d.sin) / a.max_speed
        d.cmd = [fwd, side, up, 0.0]
        return d.cmd

    def _fence(self, d: Drone):
        """The setpoint command back inside the boundary when the agent would leave it within FENCE_S, else None.
        Leaving it ends Unity's mission and freezes both drones."""
        sc = self.scene
        ahead = d.pos + d.vel * FENCE_S            # the boundary top (about 83 m) counts too
        if np.all(ahead >= sc.lo) and np.all(ahead <= sc.hi):
            return None
        inset = (sc.hi[:2] - sc.lo[:2]) * 0.1
        tgt = d.pos.copy()
        tgt[:2] = np.clip(tgt[:2], sc.lo[:2] + inset, sc.hi[:2] - inset)
        tgt[2] = np.clip(tgt[2], self.args.min_alt, sc.hi[2] - 15.0)
        target, d.target = d.target, (tgt, np.zeros(3))
        try:
            return self._command(d)
        finally:
            d.target = target

    def _unstick(self, d: Drone):
        """A full climb for STUCK_S / 2 when the agent's drone has moved under 0.5 m in STUCK_S, else None.
        The agent does not climb off what it rests on, and contact no longer ends the mission (FLOODUAV_SOFT)."""
        t = self.steps * SIM_DT
        if t < self._climb_until:
            return [0.0, 0.0, 1.0, 0.0]
        t0, p0 = self._still
        if t < t0 or np.linalg.norm(d.pos - p0) > 0.5:
            self._still = (t, d.pos.copy())
        elif t - t0 > STUCK_S:
            self._climb_until, self._still = t + STUCK_S / 2, (t + STUCK_S / 2, d.pos.copy())
            print(f"agent drone stuck at {np.round(d.pos, 1).tolist()}: climbing", flush=True)
            return [0.0, 0.0, 1.0, 0.0]
        return None

    def run(self):
        wall0, sim0 = time.monotonic(), 0
        while True:
            with self.lock:
                acts = [self._command(d) for d in self.drones]
                if self.pilot == "onnx":
                    d0 = self.drones[0]
                    over = self._fence(d0)
                    if over is None:
                        over = self._unstick(d0)
                    acts[0] = d0.cmd = over if over is not None else self.flying.act(self.obs)
                limits = self._search_limits()
                reset, self.pending_reset = self.pending_reset, False
                layout, self.pending_layout = self.pending_layout, None
            if limits != self._limits_sent:
                self.fleet.send([7, *limits])
                self._limits_sent = limits
            if reset:
                self.env.reset()
                self._reload_info()
                with self.lock:
                    self.found = {}
                    self._land_all()
                    self._limits_sent = None
                if self.layout_sent:
                    self.layout_sent = False
                    self.layout_done.set()
            if layout is not None:
                # Unity takes the layout with this step; the mission restarts with it on the next one
                self.fleet.send([6, *layout])
                with self.lock:
                    self.layout_sent, self.pending_reset = True, True
            self.env.set_actions(self.name, ActionTuple(continuous=np.clip(np.array([acts[0]], np.float32), -1, 1)))
            if len(acts) > 1:
                self.fleet.send([1, *acts[1]])
            self.env.step()
            self.steps += 1
            self._read_obs()
            ahead = (self.steps - sim0) * SIM_DT / self.args.rate - (time.monotonic() - wall0)
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -2.0:          # Unity can't keep up: don't try to catch up later
                wall0, sim0 = time.monotonic(), self.steps

    def camera(self, msg):
        if msg.get("view") == "follow":
            self.fleet.send([4, 0, msg["distance"], msg["yaw"], msg["pitch"], int(msg.get("drone", 0))])
        else:
            self.fleet.send([4, 1 + int(msg["drone"]), msg["fov"], msg["pan"], msg["tilt"]])

    def map_jpg(self, size, timeout=30.0):
        """Unity renders the map on its next step and writes the file; None if it does not come."""
        path = Path(self.args.map_file)
        path.unlink(missing_ok=True)
        self.fleet.send([5, size])
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if path.exists():
                return path.read_bytes()
            time.sleep(0.2)
        return None


class Grabber:
    """One ffmpeg x11grab per screen region, started on first use; keeps the latest JPEG."""

    def __init__(self, display: str, fps: int):
        self.display, self.fps = display, fps
        self.frames: dict[str, bytes] = {}
        self.count: dict[str, int] = {}
        self.cv = threading.Condition()
        self.started = set()

    def ensure(self, name: str):
        with self.cv:
            if name in self.started:
                return
            self.started.add(name)
        threading.Thread(target=self._run, args=(name,), daemon=True).start()

    def _run(self, name):
        region, small = name.partition("@")[::2]        # "<region>@small": scaled down, fewer frames
        x, y, w, h = REGIONS[region]
        fps = min(self.fps, SMALL_FPS) if small else self.fps
        scale = ["-vf", f"scale={min(w, SMALL_W)}:-2"] if small else []
        cmd = ["ffmpeg", "-loglevel", "error", "-f", "x11grab", "-framerate", str(fps), "-video_size", f"{w}x{h}",
               "-i", f"{self.display}+{x},{y}", *scale, "-f", "mjpeg", "-q:v", "5", "pipe:1"]
        while True:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            buf = b""
            while chunk := p.stdout.read1(65536):     # what is there: a full read waits for 64 kB
                buf += chunk
                while (s := buf.find(b"\xff\xd8")) >= 0 and (e := buf.find(b"\xff\xd9", s)) >= 0:
                    with self.cv:
                        self.frames[name] = buf[s:e + 2]
                        self.count[name] = self.count.get(name, 0) + 1
                        self.cv.notify_all()
                    buf = buf[e + 2:]
            p.wait()
            time.sleep(1.0)

    def latest(self, name, timeout=5.0, after=-1):
        """(frame number, JPEG) of the first frame after frame number `after`."""
        self.ensure(name)
        with self.cv:
            self.cv.wait_for(lambda: self.count.get(name, 0) > after, timeout)
            return self.count.get(name, 0), self.frames.get(name)


class H264:
    """One ffmpeg x11grab and H.264 encoder per screen region while anyone watches it, shared by its viewers.
    NVENC on the GPU, libx264 when NVENC fails. A new or lagging viewer waits for the next key frame (one a second)."""

    ENCODERS = (["h264_nvenc", "-preset", "p1", "-tune", "ll", "-bf", "0", "-aud", "1"],
                ["libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-x264-params", "aud=1"])

    def __init__(self, display: str, fps: int):
        self.display, self.fps = display, fps
        self.lock = threading.Lock()
        self.viewers: dict[str, list[dict]] = {}
        self.enc = 0

    def watch(self, name) -> dict:
        """A viewer of region `name`: its queue holds (key frame, access unit)."""
        v = {"q": queue.Queue(maxsize=2 * self.fps), "key": True}
        with self.lock:
            start = name not in self.viewers
            self.viewers.setdefault(name, []).append(v)
        if start:
            threading.Thread(target=self._run, args=(name,), daemon=True).start()
        return v

    def leave(self, name, v):
        with self.lock:
            self.viewers[name].remove(v)

    def _send(self, name, au):
        key = any(m.group(1)[0] & 0x1F == 5 for m in NAL.finditer(au))
        with self.lock:
            for v in self.viewers[name]:
                if v["key"] and not key:
                    continue
                try:
                    v["q"].put_nowait((key, au))
                    v["key"] = False
                except queue.Full:          # it falls behind: skip to the next key frame
                    v["key"] = True

    def _run(self, name):
        x, y, w, h = REGIONS[name]
        while True:
            with self.lock:
                if not self.viewers[name]:          # nobody watches: stop, the next viewer starts it again
                    del self.viewers[name]
                    return
            enc = self.ENCODERS[self.enc]
            cmd = ["ffmpeg", "-loglevel", "error", "-f", "x11grab", "-framerate", str(self.fps), "-video_size", f"{w}x{h}",
                   "-i", f"{self.display}+{x},{y}", "-pix_fmt", "yuv420p", "-c:v", *enc, "-g", str(self.fps),
                   "-b:v", "2M", "-maxrate", "2M", "-bufsize", "1M", "-bsf:v", "dump_extra", "-f", "h264", "pipe:1"]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            buf, sent, stopped = b"", False, False
            while chunk := p.stdout.read1(65536):
                buf += chunk
                while (e := buf.find(AUD, 4)) > 0:         # an access unit is complete when the next one starts
                    self._send(name, buf[:e])
                    buf, sent = buf[e:], True
                with self.lock:
                    stopped = not self.viewers[name]
                if stopped:
                    p.kill()
            p.wait()
            if stopped:
                continue
            if not sent and self.enc + 1 < len(self.ENCODERS):
                print(f"H.264 encoder {enc[0]} failed for {name}; trying {self.ENCODERS[self.enc + 1][0]}", flush=True)
                self.enc += 1
            elif not sent:
                time.sleep(1.0)


def make_handler(unity: Unity, grab: Grabber, h264: H264, args):
    sc = unity.scene
    maps: dict[str, bytes] = {}       # the map render, made once: the town does not change

    def agent_list():
        found = unity.agents()
        return {"onnx": bool(found), "agent": unity.agent,
                "agents": [{"id": k, "label": a["label"]} for k, a in found.items()]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/time":
                return self._json({"sim_time": unity.steps * SIM_DT})
            if path == "/state":
                return self._json(unity.snapshot())
            if path == "/agents":
                return self._json(agent_list())
            if path == "/info":
                return self._json({"area_en": sc.area(), "overview_en": sc.overview_box(), "home": unity.home.tolist(),
                                   "homes": [d.home.round(2).tolist() for d in unity.drones],
                                   "camera_fov": sc.info.get("camera_fov", 60.0),
                                   "survivors": [{"id": k, "e": v[0], "n": v[1]} for k, v in sc.survivors().items()],
                                   "kinds": {s["id"]: s.get("kind", "fixed") for s in sc.info["survivors"]},
                                   "cells": sc.cells(),
                                   "survivor_seed": sc.info.get("survivor_seed", -1),
                                   "survivor_shares": sc.info.get("survivor_shares"),
                                   "limits": {"lo": sc.lo.tolist(), "hi": sc.hi.tolist()},
                                   "min_alt": args.min_alt, "max_speed": args.max_speed, "lead": args.lead,
                                   **agent_list()})
            if path == "/map.jpg":
                if "map" not in maps:
                    jpg = unity.map_jpg(args.map_px)
                    if jpg is None:
                        return self._json({"error": "no map render"}, 503)
                    maps["map"] = jpg
                jpg = maps["map"]
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
                return
            if path == "/overview.jpg":
                _, jpg = grab.latest("overview")
                if jpg is None:
                    return self._json({"error": "no frame yet"}, 503)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
                return
            if path.startswith("/h264/") and (name := path[len("/h264/"):]) in REGIONS:
                v = h264.watch(name)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "video/h264")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    while True:
                        try:
                            key, au = v["q"].get(timeout=5.0)
                        except queue.Empty:
                            continue
                        self.wfile.write(struct.pack(">IB", len(au), key) + au)
                except (BrokenPipeError, ConnectionResetError):
                    return
                finally:
                    h264.leave(name, v)
            if path.startswith("/stream/") and (name := path[len("/stream/"):]) in REGIONS:
                if "small=1" in self.path:
                    name += "@small"
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while True:
                        k, jpg = grab.latest(name, after=last)
                        if jpg is None or k == last:
                            continue
                        last = k
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            msg = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/setpoint":
                k = int(msg.get("drone", 0))
                if msg.get("land"):
                    unity.set_target(k, mode="land")
                elif msg.get("hold"):
                    unity.set_target(k, mode="hold")
                else:
                    unity.set_target(k, msg["pos"], msg.get("vel"))
                return self._json({"ok": True})
            if self.path == "/speed":
                unity.set_speed(msg["speed"])
                return self._json({"ok": True, "speed": unity.speed_cap})
            if self.path == "/pilot":
                try:
                    unity.set_pilot(msg.get("pilot", "setpoint"), msg.get("limits"), msg.get("agent"))
                except (ValueError, ImportError, OSError) as exc:
                    return self._json({"error": str(exc)}, 400)
                return self._json({"ok": True, "pilot": unity.pilot, "limits": unity.limits, "agent": unity.agent})
            if self.path == "/reset":
                unity.reset()
                return self._json({"ok": True})
            if self.path == "/survivors":
                seed = msg.get("seed")
                seed = int(np.random.default_rng().integers(0, 1 << 24)) if seed is None else int(seed)
                try:
                    shares = parse_shares(msg.get("shares"))
                except ValueError as exc:
                    return self._json({"error": str(exc)}, 400)
                if not unity.new_layout(seed, int(msg.get("count", -1)), shares):
                    return self._json({"error": "Unity did not restart with the new layout"}, 504)
                return self._json({"ok": True, "seed": seed, "count": len(sc.survivors())})
            if self.path == "/camera":
                unity.camera(msg)
                return self._json({"ok": True})
            return self._json({"error": "not found"}, 404)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="unity/SearchPickupPatched/FloodUAV.x86_64")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--worker", type=int, default=30, help="ML-Agents worker id (port 5005 + id)")
    ap.add_argument("--rate", type=float, default=2.0, help="sim seconds per wall second (upper bound)")
    ap.add_argument("--time-scale", type=float, default=2.0,
                    help="Unity time scale; keep it near --rate: above it, every rendered frame waits on many "
                         "Python steps and the video drops to a few frames per second")
    ap.add_argument("--quality", type=int, default=2)
    ap.add_argument("--unity-args", default="-force-vulkan",
                    help="extra player arguments; Vulkan renders on the NVIDIA GPU under Xvfb (OpenGL falls back to llvmpipe)")
    ap.add_argument("--max-speed", type=float, default=15.0, help="m/s, horizontal")
    ap.add_argument("--climb", type=float, default=4.0, help="m/s, vertical")
    ap.add_argument("--tilt", type=float, default=25.0, help="degrees; sets the horizontal acceleration")
    ap.add_argument("--wing-area", type=float, default=1.0, help="m2; the build's 100 caps the speed near 2.5 m/s")
    ap.add_argument("--kp", type=float, default=0.5, help="position gain, 1/s")
    ap.add_argument("--brake", type=float, default=1.5, help="planned braking deceleration, m/s2")
    ap.add_argument("--lead", type=float, default=1.5,
                    help="s; the setpoint controller steers from the drone's position this far ahead, which damps the swing")
    ap.add_argument("--min-alt", type=float, default=40.0, help="m above the start point; below about 15 m it hits bridges")
    ap.add_argument("--max-alt", type=float, default=80.0)
    ap.add_argument("--margin", type=float, default=0.03, help="fraction of the boundary kept clear (leaving it ends the mission)")
    ap.add_argument("--follow", default="0,30,-30", help="follow camera offset behind the drone, m (about 35 degrees down)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":97"))
    ap.add_argument("--info", default="/tmp/flooduav_info.json")
    ap.add_argument("--unity-log", default="/tmp/flooduav_unity.log")
    ap.add_argument("--map-file", default="/tmp/flooduav_map.jpg", help="where Unity writes the map render")
    ap.add_argument("--survivor-seed", type=int, default=-1,
                    help="survivor layout seed at startup; -1 keeps the build's fixed survivors (patch Layout.cs)")
    ap.add_argument("--survivor-count", type=int, default=-1,
                    help="number of survivors (1-100); -1 keeps the build's 15; more than 15 needs --survivor-seed")
    ap.add_argument("--map-px", type=int, default=4096, help="map render size, pixels per side")
    ap.add_argument("--models", default=str(ROOT / "models"),
                    help="drop-in folder of trained search agents (sim/flooduav/agents.py), flown with POST /pilot")
    ap.add_argument("--agent", default="unity_survivor_search/survivor_search_ppo_19m",
                    help="the default agent: its path under --models, without .onnx")
    args = ap.parse_args()

    unity = Unity(args)
    grab = Grabber(args.display, args.fps)
    grab.ensure("overview")      # the UI asks for the map image right away; ffmpeg takes a few seconds to start
    threading.Thread(target=unity.run, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(unity, grab, H264(args.display, args.fps), args))
    print(f"FloodUAV bridge on http://{args.host}:{args.port}, start {unity.home.round(1).tolist()}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
