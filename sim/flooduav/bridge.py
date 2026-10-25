"""FloodUAV bridge: runs the patched Unity build and serves it to the mission server.

The mission server (sar/mission/flooduav.py) talks to this bridge the way it talks to the
Isaac bridge: sim time, video streams and a follow-view call, plus a drone state and a
setpoint endpoint for the two Unity drones. See sim/flooduav/README.md.

  GET  /time               {"sim_time": s}
  GET  /info               scene info: boundary, start, survivors, area polygon, map image box (mission meters)
  GET  /state              drone poses in mission meters, episode, found survivors
  POST /setpoint           {"drone": k, "pos": [e, n, u], "vel": [ve, vn, vu]}; "hold": true holds where it is,
                           "land": true descends and rests on the ground below
  POST /speed              {"speed": m/s} horizontal speed cap for both drones
  POST /reset              restart Unity's mission: drones back on their pads, survivors unfound
  POST /view               {"drone": k} follow camera target
  POST /camera             {"view": "follow", "distance", "yaw", "pitch"} or {"view": "drone", "drone": k, "fov", "pan", "tilt"}
  GET  /stream/follow      main Unity view
  GET  /stream/cam0, cam1  drone cameras (0 search drone, 1 the build's pickup drone)
  GET  /stream/overview    square top-down view
  GET  /overview.jpg       one top-down frame from the live stream
  GET  /map.jpg            one high resolution top-down render (--map-px), for the map background

Mission frame: east = boundary +x, north = boundary +z, up = boundary +y, in meters, with the
origin at the drone start point. Run with .venv-fly and DISPLAY set (sim/flooduav/run.sh).
"""

import argparse
import json
import math
import os
import re
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

# observation indices (SurvivorSearchAgent.CollectObservations)
VEL = slice(0, 3)          # local velocity / (10, 3, 10)
AGL_HIT, AGL = 9, 10       # down ray hit flag, distance / 100 m
POS = slice(11, 14)        # position in the boundary, normalized to [-1, 1]
SIN, COS = 14, 15          # heading from boundary forward (north), clockwise
RAYS = slice(16, 24)       # 8 horizontal obstacle rays / 30 m, from forward clockwise
UP_RAY, DOWN_RAY = 24, 25
SIM_DT = 0.1               # one decision = 5 physics steps of 0.02 s

# screen layout from Hook.SplitViews: main view, the two drone cameras stacked, square overview
W, H = 2640, 720
REGIONS = {"follow": (0, 0, 1280, 720), "cam0": (1280, 0, 640, 360), "cam1": (1280, 360, 640, 360),
           "overview": (1920, 0, 720, 720)}
RECORDED = re.compile(r"Recorded (\S+) at")
FINISHED = re.compile(r"Mission (Failed|TimedOut|Succeeded): (.*)")


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
        os.environ.update({
            "FLOODUAV_HSPEED": str(args.max_speed), "FLOODUAV_VSPEED": str(args.climb),
            "FLOODUAV_TILT": str(args.tilt), "FLOODUAV_TIME_LIMIT": "1000000",
            "FLOODUAV_SPLIT": "1", "FLOODUAV_INFO": args.info, "FLOODUAV_FLEET": "1", "FLOODUAV_SOFT": "1",
            "FLOODUAV_FOLLOW": args.follow, "FLOODUAV_AIRCRAFT": f"wingArea={args.wing_area}",
            "FLOODUAV_MAP": args.map_file,
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
            if not ended:
                return
            self.env.step()          # ML-Agents starts the next episode; the drones are back on their pads

    def _land_all(self):
        for d in self.drones:
            d.mode, d.target, d.hold_at = "land", None, None

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
                "speed_cap": self.speed_cap, "drones": [d.snapshot() for d in self.drones],
                "found": [{"id": k, "e": v[0], "n": v[1]} for k, v in self.found.items()],
            }

    # ---------------------------------------------------------------- control
    def set_target(self, k, pos=None, vel=None, mode="fly"):
        with self.lock:
            d = self.drones[k]
            d.mode, d.hold_at = mode, d.pos.copy()
            d.target = None if pos is None else (np.asarray(pos, float), np.asarray(vel if vel is not None else [0, 0, 0], float))

    def set_speed(self, v):
        with self.lock:
            self.speed_cap = float(np.clip(v, 1.0, self.args.max_speed))

    def reset(self):
        """Restart Unity's mission: drones back on their pads, survivors unfound."""
        with self.lock:
            self.pending_reset = True

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
        dist = float(np.hypot(*err[:2]))
        cap = 3.0 if landing else self.speed_cap
        v = ff[:2] + (err[:2] / dist * min(a.kp * dist, math.sqrt(2 * a.brake * dist)) if dist > 1e-3 else 0.0)
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

    def run(self):
        wall0, sim0 = time.monotonic(), 0
        while True:
            with self.lock:
                acts = [self._command(d) for d in self.drones]
                reset, self.pending_reset = self.pending_reset, False
            if reset:
                self.env.reset()
                with self.lock:
                    self.found = {}
                    self._land_all()
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

    def view(self, k):
        self.fleet.send([2, k])

    def camera(self, msg):
        if msg.get("view") == "follow":
            self.fleet.send([4, 0, msg["distance"], msg["yaw"], msg["pitch"]])
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
        x, y, w, h = REGIONS[name]
        cmd = ["ffmpeg", "-loglevel", "error", "-f", "x11grab", "-framerate", str(self.fps), "-video_size", f"{w}x{h}",
               "-i", f"{self.display}+{x},{y}", "-f", "mjpeg", "-q:v", "5", "pipe:1"]
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


def make_handler(unity: Unity, grab: Grabber, args):
    sc = unity.scene
    maps: dict[str, bytes] = {}       # the map render, made once: the town does not change

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
            if path == "/info":
                return self._json({"area_en": sc.area(), "overview_en": sc.overview_box(), "home": unity.home.tolist(),
                                   "homes": [d.home.round(2).tolist() for d in unity.drones],
                                   "camera_fov": sc.info.get("camera_fov", 60.0),
                                   "survivors": [{"id": k, "e": v[0], "n": v[1]} for k, v in sc.survivors().items()],
                                   "limits": {"lo": sc.lo.tolist(), "hi": sc.hi.tolist()},
                                   "min_alt": args.min_alt, "max_speed": args.max_speed})
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
            if path.startswith("/stream/") and (name := path[len("/stream/"):]) in REGIONS:
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
            if self.path == "/reset":
                unity.reset()
                return self._json({"ok": True})
            if self.path == "/view":
                unity.view(int(msg.get("drone", 0)))
                return self._json({"ok": True})
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
    ap.add_argument("--min-alt", type=float, default=40.0, help="m above the start point; below about 15 m it hits bridges")
    ap.add_argument("--max-alt", type=float, default=80.0)
    ap.add_argument("--margin", type=float, default=0.03, help="fraction of the boundary kept clear (leaving it ends the mission)")
    ap.add_argument("--follow", default="0,30,-30", help="follow camera offset behind the drone, m (about 35 degrees down)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":97"))
    ap.add_argument("--info", default="/tmp/flooduav_info.json")
    ap.add_argument("--unity-log", default="/tmp/flooduav_unity.log")
    ap.add_argument("--map-file", default="/tmp/flooduav_map.jpg", help="where Unity writes the map render")
    ap.add_argument("--map-px", type=int, default=4096, help="map render size, pixels per side")
    args = ap.parse_args()

    unity = Unity(args)
    grab = Grabber(args.display, args.fps)
    grab.ensure("overview")      # the UI asks for the map image right away; ffmpeg takes a few seconds to start
    threading.Thread(target=unity.run, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(unity, grab, args))
    print(f"FloodUAV bridge on http://{args.host}:{args.port}, start {unity.home.round(1).tolist()}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
