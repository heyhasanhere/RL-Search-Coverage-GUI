"""Mission orchestrator (plan section 4).

Owns the world model (area grid, belief, looks), runs one behavior task per
drone (search, rescue, standby, return, abort), and a tick loop for the
flight-time battery rule, looks and detections, confirmation, rescue dispatch,
standby replacement, and auto-staging. The search policy only picks segments.
"""

import asyncio
import base64
import contextlib
import io
import json
import math
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from sar.config import MissionConfig
from sar.mission.flight import FlightLayer
from sar.mission.geo import AreaGrid, Frame, koester_prior, points_in_polygon
from sar.mission.search_policy import MODES, AgentView, SearchPolicy, grid_for_extent
from sar.mission.search_policy_v2 import MODES_V2, get_policy_v2
from sar.mission.vehicle import SimClock, Vehicle

AIRBORNE = ("SEARCH", "RESCUE_TRANSIT", "ON_SURVIVOR", "RETURNING", "ABORT")
TRANSIT_LOOKS = ("RESCUE_TRANSIT", "ON_SURVIVOR", "RETURNING", "STANDBY")   # non-search flights whose camera counts
STOP_FRACTION = 0.02          # search ends when this fraction of the prior remains
LEAD_SPEED = 3.0              # m/s the carrot may run ahead of the drone's speed along the leg
# The sim's own trained agent flies the drone (vehicles with agent_capable, FloodUAV's search drone);
# the other search drones use rl. The orchestrator only counts the agent's looks and finds.
AGENT_MODE = "rl_unity"
AGENT_LOOK_AGL = 10.0         # m above the ground below; lower, the agent's camera look does not count
TAKEOFF_S = 10.0              # s added to a landed drone's arrival estimate


@dataclass
class Drone:
    id: int
    name: str
    role: str
    vehicle: Vehicle
    state: str = "IDLE"
    battery_s: float = 0.0
    task: asyncio.Task | None = None
    task_text: str = ""
    path: list = field(default_factory=list)      # queued 3D points
    carrot: np.ndarray | None = None
    carrot_v: float = 0.0                         # m/s the carrot moves at, ramped by MissionConfig.accel
    segment: list = field(default_factory=list)   # current segment cells
    plan_cell: tuple | None = None                # end cell of the queued path
    heading: int = -1
    last_cell: tuple | None = None
    confirming: bool = False
    relook: dict | None = None
    target: np.ndarray | None = None
    standby_point: np.ndarray | None = None
    survivor: int | None = None
    entry: tuple | None = None
    agent: bool = False                           # the sim's trained agent flies it (AGENT_MODE)
    view: set = field(default_factory=set)        # cells in the camera footprint at the last look
    resume_search: bool = False                   # switched from search by _auto_switch: search again after the rescue


@dataclass
class Survivor:
    id: int
    e: float
    n: float
    score: float
    t_detected: float
    views: list = field(default_factory=list)     # viewpoints (e, n) that saw it
    status: str = "candidate"
    snapshot: bytes | None = None
    assigned: int | None = None
    t_confirmed: float | None = None
    t_reached: float | None = None


class Mission:
    def __init__(self, cfg: MissionConfig, frame: Frame, grid: AreaGrid, prior, expected, mode, lkp_en, t0):
        self.grid, self.mode, self.lkp_en, self.t0 = grid, mode, lkp_en, t0
        self.expected = expected
        self.belief = prior * expected
        self.belief0 = float(self.belief.sum())
        self.looks = np.zeros(grid.mask.shape, np.int32)
        self.status = "running"
        self.stopping = False
        hc = grid.cell_of(0.0, 0.0)
        self.home_cell = (int(np.clip(hc[0], 0, grid.g - 1)), int(np.clip(hc[1], 0, grid.g - 1)))


class Orchestrator:
    def __init__(self, cfg: MissionConfig, vehicles: list[Vehicle], roles: list[str], clock: SimClock,
                 ckpt_root: Path | None, detector_factory, log_path: Path | None = None, seed: int = 0,
                 flight: FlightLayer | None = None, sensors=None):
        self.cfg, self.clock = cfg, clock
        # flight layer: mission heights are AGL; it maps them to each vehicle's home-relative frame
        self.flight = flight or FlightLayer(None)
        self.sensors = sensors      # optional: .latest(vid) -> (range_m, depth) from the sim bridge
        # optional: (grid, lkp_en, lkp_time_h, scenario, water_level_m) -> {key: [g,g] map}; the first entry
        # with key "blend" (or the sum of all entries) is the prior. None: the Koester ring prior.
        self.prior_builder = None
        self.frame = Frame(*cfg.home)
        # D1, D2, ...: any drone can search or rescue; the role only picks the default task
        self.drones = [Drone(v.id, f"D{i + 1}", role, v, battery_s=cfg.battery_s)
                       for i, (v, role) in enumerate(zip(vehicles, roles))]
        self.ckpt_root, self.detector_factory, self.seed = ckpt_root, detector_factory, seed
        self.mission: Mission | None = None
        self.detector = None
        self.truth_en = None
        self.policies: dict[int, SearchPolicy] = {}
        self._warm_lock = threading.Lock()       # warm_policies runs in a worker thread
        self._warm_grids: set[int] = set()
        self._link_grace = -1e9                  # no lost-link check before this time (after a stall)
        self.survivors: dict[int, Survivor] = {}
        self.options = {"auto_dispatch": True, "auto_replace": True, "auto_stage": True, "auto_switch": True,
                        "relook": False}           # relook: fly a re-look before confirming a candidate
        self.events: list[dict] = []
        self.subscribers: list[asyncio.Queue] = []
        self._next_sid = 1          # survivor numbers start at 1 in every mission
        self._last_stage = -1e9
        self._loop_task = None
        self._last_track = -1e9      # replay log timestamps (plan 11.2)
        self._last_belief_log = -1e9
        self._log = open(log_path, "a") if log_path else None
        self.metrics = {"looks": 0, "relooks": 0, "false_dispatch": 0}
        self.last_start: dict | None = None      # start_mission arguments, for restart
        self.on_restart = None                   # optional async callback: reset the sim world
        self.on_new_survivors = None             # optional async callback(count, mix): new survivor layout, returns (seed, count)
        self.drone_speed: float | None = None    # operator speed cap, m/s; None: vehicle default
        self.on_graphics = None                  # optional async callback(preset): sim graphics preset, returns the bridge's graphics
        self.graphics: dict | None = None        # the sim's graphics: preset, auto, scale, msaa, fps, bitrate, presets
        # AGENT_MODE flight limits: "bridge" (the operator's speed cap) or "training" (the agent's own, speed cap off)
        self.agent_limits = "bridge"
        # AGENT_MODE trained agents ({"id", "label"}) and the one that flies; set by the backend
        self.agents: list[dict] = []
        self.agent: str | None = None
        self.refresh_agents = None               # optional async callback: rescan the agents (drop-in models)
        self.modes = tuple(MODES) + ((AGENT_MODE,) if any(getattr(v, "agent_capable", False) for v in vehicles) else ())

    # ---------------------------------------------------------------- basics
    async def start(self):
        for d in self.drones:
            await d.vehicle.start()
        self._loop_task = asyncio.create_task(self._loop())

    async def shutdown(self):
        if self._loop_task:
            self._loop_task.cancel()
        for d in self.drones:
            if d.task:
                d.task.cancel()
            await d.vehicle.stop()
        if self._log:
            self._log.close()

    def now(self):
        return self.clock.now()

    def t_mission(self):
        return self.now() - self.mission.t0 if self.mission else 0.0

    def event(self, text, level="info"):
        ev = {"type": "event", "t": round(self.t_mission(), 1), "level": level, "text": text}
        self.events = self.events[-199:] + [ev]
        self._publish(ev)
        self.log("event", **ev)

    def log(self, kind, **kw):
        if self._log:
            self._log.write(json.dumps({"kind": kind, "t": round(self.now(), 3), **kw}) + "\n")

    # ------------------------------------------------- replay logging (plan 11.2)
    # The mission log is what the split replay reads (sar/mission/replay.py).
    # These rows are written from the existing log writer only; they do not
    # change what the mission does.
    def log_track(self):
        """One row per sample: where the drones are and how much is found."""
        if not self._log or not self.mission:
            return
        # The replay rebuilds the live panels from these rows, so they carry the
        # same drone and mission fields as the state message.
        msg = self.state_msg()
        keep = ("id", "name", "role", "state", "pos", "alt", "heading", "battery_s", "battery_frac",
                "segment", "target", "standby_point", "task", "level", "alt_agl", "sortie", "swap_remaining")
        self.log("track",
                 found=round(self.expected_found() / max(self.mission.expected, 1e-9), 5),
                 found_abs=round(self.expected_found(), 4),
                 covered=msg["mission"]["covered_fraction"], remaining=msg["mission"]["remaining_fraction"],
                 lkp=msg["mission"]["lkp"],
                 drones=[{k: d[k] for k in keep} for d in msg["drones"]])

    def log_belief(self):
        """A belief snapshot for the replay heat map (subsampled in replay.py)."""
        if not self._log or not self.mission:
            return
        msg = self.belief_msg()
        self.log("belief", corners=msg["corners"], png=msg["png"], coverage_png=msg["coverage_png"])

    def _publish(self, msg):
        for q in self.subscribers:
            if q.qsize() < 100:
                q.put_nowait(msg)

    def lonlat(self, e, n):
        lon, lat = self.frame.to_lonlat(e, n)
        return [round(float(lon), 7), round(float(lat), 7)]

    def en(self, lonlat):
        e, n = self.frame.to_en(lonlat[0], lonlat[1])
        return np.array([float(e), float(n)])

    def _set_task(self, d: Drone, coro, state: str, text: str):
        if d.task and not d.task.done() and d.task is not asyncio.current_task():
            d.task.cancel()
        d.state, d.task_text = state, text
        d.path, d.carrot, d.relook, d.confirming = [], None, None, False
        if state != "RESCUE_TRANSIT":
            d.resume_search = False
        if state != "SEARCH":
            d.segment, d.plan_cell, d.entry = [], None, None
        d.task = asyncio.create_task(self._guard(d, coro))
        self.log("state", drone=d.id, state=state, text=text)

    async def _guard(self, d, coro):
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # keep the orchestrator alive; report the failure
            self.event(f"{d.name} task failed: {exc!r}", "error")
            traceback.print_exc()
            if not d.vehicle.tel.in_air:
                d.state, d.task_text = "LANDED", "task error on the ground"
                self.log("state", drone=d.id, state="LANDED", text=repr(exc))
            elif d.state == "ABORT":
                # the abort return itself failed: leave the drone to PX4's own failsafes
                d.state, d.task_text = "FAILED", "return failed"
                self.log("state", drone=d.id, state="FAILED", text=repr(exc))
            else:
                d.state, d.task_text = "ABORT", "task error, returning"
                self.log("state", drone=d.id, state="ABORT", text=repr(exc))
                d.task = asyncio.create_task(self._guard(d, self._abort(d)))

    # -------------------------------------------------------------- missions
    def start_mission(self, polygon=None, lkp=None, mode="rl", n_search=2, expected_survivors=3.0,
                      options=None, mask=None, prior=None, truth_en=None, scenario="wilderness",
                      lkp_time_h=None, regions=None, water_level_m=0.0):
        cfg = self.cfg
        if self.mission and self.mission.status == "running":
            raise ValueError("a mission is already running")
        self.last_start = dict(polygon=polygon, lkp=lkp, mode=mode, n_search=n_search,
                               expected_survivors=expected_survivors, options=options, mask=mask, prior=prior,
                               truth_en=truth_en, scenario=scenario, lkp_time_h=lkp_time_h, regions=regions,
                               water_level_m=water_level_m)
        if mode not in self.modes:
            raise ValueError(f"mode must be one of {self.modes}")
        if mask is not None:
            g = mask.shape[0]
            grid = AreaGrid.from_mask(self.frame, mask, cfg.cell_m, center_en=(0.0, 0.0))
        else:
            g = self._grid_size(polygon)
            grid = AreaGrid(self.frame, polygon, cfg.cell_m, g)
        lkp_en = self.en(lkp) if lkp is not None else None
        layers = {}
        if prior is None:
            if self.prior_builder is not None:
                layers = self.prior_builder(grid, lkp_en, lkp_time_h, scenario, float(water_level_m))
            else:
                layers = {"koester": koester_prior(grid, lkp_en)}
            prior = self._blend(grid, layers, regions)
        self.policy = self._policy(g)
        self.mission = Mission(cfg, self.frame, grid, prior, float(expected_survivors), mode, lkp_en, self.now())
        # compile every mode now: a compile during flight would stall setpoint streaming
        for md in MODES:
            if md == "rl" and self.ckpt_root is None:
                continue
            self.mission.mode = md
            self._decide(self.drones[0], (grid.g // 2, grid.g // 2), warm=True)
        self.mission.mode = mode
        self.mission.t0 = self.now()
        for d in self.drones:          # the compile above may have taken longer than the link timeout
            d.vehicle.tel.t_update = max(d.vehicle.tel.t_update, self.now())
        self.mission.scenario, self.mission.prior_layers = scenario, layers
        self.mission.polygon = [self.lonlat(*p) for p in grid.poly_en] if polygon is None else [list(p) for p in polygon]
        self.mission.lkp = lkp
        if options:
            self.options.update({k: bool(v) for k, v in options.items() if k in self.options})
        self.survivors.clear()
        self._next_sid = 1
        self.truth_en = truth_en
        self.detector = self.detector_factory(self, truth_en)
        self.metrics = {"looks": 0, "relooks": 0, "false_dispatch": 0}
        searchers = [d for d in self.drones if d.role == "search"][: int(n_search)]
        if len(searchers) < int(n_search):
            searchers += [d for d in self.drones if d.role != "search"][: int(n_search) - len(searchers)]
        # The mission row comes first: it starts the mission in the replay log
        # (plan 11.2), so every later row of this mission belongs to it.
        self.log("mission", mode=mode, g=g, cells=int(grid.mask.sum()), expected=expected_survivors,
                 scenario=scenario, seed=self.seed, polygon=self.mission.polygon)
        for d in self.drones:
            if d in searchers:
                self._set_task(d, self._search(d), "SEARCH", "taking off")
            elif d.state in ("IDLE", "LANDED"):
                d.standby_point = d.vehicle.home[:2].copy()
                d.state, d.task_text = "STANDBY", "landed at home"
        self.event(f"mission started: {mode}, {len(searchers)} search drones, grid {g}x{g}, "
                   f"{int(grid.mask.sum())} cells")
        if self._log:
            with contextlib.suppress(Exception):
                self.log("prior", **{k: v for k, v in self.prior_msg().items() if k != "type"})

    def stop_mission(self, reason="operator stop"):
        m = self.mission
        if not m or m.status != "running":
            return
        m.stopping = True
        for d in self.drones:
            if d.state in AIRBORNE and d.state not in ("RETURNING", "ABORT"):
                self._set_task(d, self._return(d, reason), "RETURNING", reason)
            elif d.state == "STANDBY" and d.vehicle.tel.in_air:
                self._set_task(d, self._return(d, reason), "RETURNING", reason)
        self.event(f"mission stopping: {reason}")

    def continue_mission(self):
        """Resume a stopped or finished mission: belief, coverage and survivors are kept, search drones go back up."""
        m = self.mission
        if not m or (m.status == "running" and not m.stopping):
            raise ValueError("no stopped mission to continue")
        m.status, m.stopping = "running", False
        n_search = int(self.last_start["n_search"]) if self.last_start else 2
        free = [d for d in self.drones if d.state != "ABORT"]
        searchers = ([d for d in free if d.role == "search"] + [d for d in free if d.role != "search"])[:n_search]
        for d in self.drones:
            if d in searchers:
                self._set_task(d, self._search(d), "SEARCH", "continuing")
            elif d.state in ("IDLE", "LANDED"):
                d.standby_point = d.vehicle.home[:2].copy()
                d.state, d.task_text = "STANDBY", "landed at home"
        self.event(f"mission continued: {len(searchers)} search drones")
        self.log("continued", searchers=[d.id for d in searchers])

    async def restart_mission(self):
        """Start the last mission again from scratch: tasks dropped, batteries full, drones on their pads."""
        if self.last_start is None:
            raise ValueError("no mission to restart")
        await self._reset_world()
        self.event("mission restarted")
        self.start_mission(**self.last_start)

    async def reset_mission(self):
        """Back to the state before the first start: no mission, no survivors, drones idle on their pads."""
        await self._reset_world()
        self.mission = None
        self.survivors.clear()
        self._next_sid = 1
        self.event("mission reset")

    async def new_survivors(self, count=None, mix=None):
        """Reset, then `count` survivors at new random positions (scene backends with a survivor layout).

        `mix` is the survivor terrain mix, weights by SHARES key (normalized); None keeps the default.
        """
        from sar.prior.flood_bank import SHARES
        if self.on_new_survivors is None:
            raise ValueError("this backend has no survivor layouts")
        if count is not None and not 1 <= int(count) <= 100:
            raise ValueError("survivor count must be 1-100")
        if mix is not None:
            if not isinstance(mix, dict) or set(mix) - set(SHARES):
                raise ValueError(f"survivor mix keys: {', '.join(SHARES)}")
            mix = {k: float(mix.get(k, 0.0)) for k in SHARES}
            if any(not math.isfinite(w) or w < 0 for w in mix.values()) or sum(mix.values()) <= 0:
                raise ValueError("survivor mix weights must be 0 or more, and not all 0")
        await self.reset_mission()
        seed, n = await self.on_new_survivors(count, mix)
        mixed = "" if mix is None else ", mix " + " ".join(
            f"{k} {100 * w / sum(mix.values()):.0f}%" for k, w in mix.items())
        self.event(f"new survivor positions: {n} survivors (seed {seed}{mixed})")

    async def _reset_world(self):
        tasks = [d.task for d in self.drones if d.task and not d.task.done()]
        for t in tasks:
            t.cancel()
        # let the old tasks finish unwinding, so none of them commands a drone of the next mission
        await asyncio.gather(*tasks, return_exceptions=True)
        for d in self.drones:
            d.task = None
        if self.mission:
            self.mission.status = "finished"
        if self.on_restart is not None:
            await self.on_restart()
        else:
            for d in self.drones:
                await d.vehicle.reset()
        for d in self.drones:
            d.state, d.task_text, d.battery_s = "IDLE", "", self.cfg.battery_s
            d.survivor, d.target, d.standby_point = None, None, None
            d.path, d.carrot, d.segment, d.plan_cell = [], None, [], None
            d.view, d.last_cell = set(), None

    def recharge(self, d: Drone):
        if d.vehicle.tel.in_air:
            raise ValueError(f"{d.name} must be landed to recharge")
        d.battery_s = self.cfg.battery_s
        self.event(f"{d.name} battery recharged")

    async def set_graphics(self, preset: str):
        if self.on_graphics is None:
            raise ValueError("this backend has no graphics presets")
        presets = (self.graphics or {}).get("presets", [])
        if presets and preset not in presets:
            raise ValueError(f"graphics preset: {', '.join(presets)}")
        g = await self.on_graphics(preset)
        self.graphics = {k: g[k] for k in ("preset", "auto", "scale", "msaa", "fps", "bitrate", "presets") if k in g}
        shown = f"auto ({g['auto']})" if preset == "auto" else preset
        self.event(f"graphics {shown}: {g['fps']} fps, {g['bitrate']}bit/s")

    async def set_drone_speed(self, v: float):
        v = float(v)
        if v <= 0:
            raise ValueError("speed must be positive")
        for d in self.drones:
            await d.vehicle.set_max_speed(v)
        self.drone_speed = v
        self.event(f"drone speed cap {v:.1f} m/s")

    # ------------------------------------------------------------ world model
    def drone_cell(self, d):
        p = d.vehicle.tel.pos
        return self.mission.grid.cell_of(p[0], p[1])

    def _agents(self, deciding: Drone, from_cell):
        g = self.mission.grid.g
        views = []
        for d in self.drones[: self.policy.n]:
            active = d.state == "SEARCH" and (d.plan_cell is not None or d is deciding)
            if d is deciding:
                cell = from_cell
                seg = []
            else:
                r, c = self.drone_cell(d)
                cell = (int(np.clip(r, 0, g - 1)), int(np.clip(c, 0, g - 1)))
                seg = list(d.segment)
            views.append(AgentView(active=active, cell=cell, heading=d.heading if d is deciding else -1,
                                   battery=d.battery_s, confirming=d.confirming, seg=seg))
        return views

    def _grid_size(self, polygon):
        e, n = self.frame.to_en(np.asarray(polygon)[:, 0], np.asarray(polygon)[:, 1])
        return grid_for_extent(int(math.ceil(max(np.ptp(e), np.ptp(n)) / self.cfg.cell_m)) + 1)

    def _policy(self, g):
        if g not in self.policies:
            self.policies[g] = SearchPolicy(g, self.ckpt_root, n_slots=3, seed=self.seed)
        return self.policies[g]

    def warm_policies(self, g):
        """Compile every search mode for grid size g on a sample state.

        It runs in a worker thread before a mission starts: a JAX compile takes up to half a
        minute, and on the event loop it would stop telemetry and trip the lost-link check.
        """
        with self._warm_lock:
            if g in self._warm_grids:
                return
            pol, c = self._policy(g), (g // 2, g // 2)
            mask = np.zeros((g, g), bool)
            mask[1:-1, 1:-1] = True
            belief = mask / mask.sum()
            agents = [AgentView(active=i == 0, cell=c, battery=self.cfg.battery_s) for i in range(pol.n)]
            args = (belief, mask, np.zeros((g, g)), agents, c, self.cfg.pd, 1.0, 0.0)
            for md in MODES:
                if md == "rl" and self.ckpt_root is None:
                    continue
                if md in MODES_V2:
                    get_policy_v2(g).decide(md, 0, *args)
                else:
                    pol.decide(md, 0, *args)
            self._warm_grids.add(g)

    def _decide(self, d: Drone, from_cell, warm=False):
        m = self.mission
        mode = "rl" if m.mode == AGENT_MODE else m.mode     # drones the agent cannot fly
        agents = self._agents(d, from_cell)
        seg_time = self.policy.cfg.ring * self.policy.cfg.t_straight
        t_frac = min(self.t_mission() / (self.policy.cfg.max_steps * seg_time), 1.0)
        slot = d.id if d.id < self.policy.n else 0
        args = (m.belief, m.grid.mask, m.looks, agents, m.home_cell, self.cfg.pd, m.belief0, t_frac)
        if mode in MODES_V2:         # mcts on env v2: action, cells and the level's altitude
            a, cells, alt = get_policy_v2(m.grid.g).decide(mode, slot, *args)
        else:
            (a, cells), alt = self.policy.decide(mode, slot, *args), self.cfg.search_alt
        if not warm:
            self.log("decide", drone=d.id, mode=mode, cell=list(from_cell), action=a)
        return a, cells, alt

    def _heading_of(self, a):
        if self.mission.mode in MODES_V2:
            return get_policy_v2(self.mission.grid.g).heading_of(a)
        return self.policy.heading_of(a)

    def _view_cells(self, d: Drone, cell):
        """Grid cells the drone's camera sees: the cell below, and every cell whose center is inside
        the vehicle's footprint box (footprint_m: half sizes, along-track on the compass heading)."""
        g = self.mission.grid
        fp = d.vehicle.footprint_m()
        cells = {tuple(cell)}
        if fp is not None:
            along, cross = fp
            p, h = d.vehicle.tel.pos, math.radians(d.vehicle.tel.yaw)
            k = int(math.ceil(math.hypot(along, cross) / self.cfg.cell_m))
            rr, cc = np.mgrid[cell[0] - k: cell[0] + k + 1, cell[1] - k: cell[1] + k + 1]
            e, n = g.cell_center(rr, cc)
            de, dn = e - p[0], n - p[1]
            a = de * math.sin(h) + dn * math.cos(h)
            x = de * math.cos(h) - dn * math.sin(h)
            inside = (np.abs(a) <= along) & (np.abs(x) <= cross)
            cells |= {(int(r), int(c)) for r, c in zip(rr[inside], cc[inside])}
        return {(r, c) for r, c in cells if g.in_grid(r, c)}

    def _look(self, d: Drone, cell, again=False):
        """One camera look: each seen cell's belief drops by pd. A cell counts once per pass under the
        footprint (cells still in view from d's last look are skipped), or again on a deliberate hold."""
        m = self.mission
        cells = self._view_cells(d, cell)
        new = cells if again else cells - d.view
        d.view = cells
        if not cells:
            return
        for r, c in new:
            if m.grid.mask[r, c]:
                m.belief[r, c] *= (1.0 - self.cfg.pd)
            m.looks[r, c] += 1
        self.metrics["looks"] += 1
        pos = d.vehicle.tel.pos.copy()
        asyncio.create_task(self._detect(d, pos, d.vehicle.tel.yaw))

    async def _detect(self, d, pos, yaw, relook=False):
        try:
            dets = await self.detector.look(d.id, pos, yaw)
        except Exception as exc:
            self.event(f"{d.name} detector error: {exc!r}", "warn")
            return []
        for det in dets:
            self._on_detection(d, det, pos)
        return dets

    def _on_detection(self, d: Drone, det, viewpoint):
        cfg = self.cfg
        for o in self.drones:   # the camera sees the other drones on their pads or flying lower
            op = o.vehicle.tel.pos
            if o is not d and op[2] < viewpoint[2] - 3.0 and math.hypot(op[0] - det.e, op[1] - det.n) < cfg.drone_mask_m:
                return
        best, bd = None, 1e9
        for s in self.survivors.values():
            dist = math.hypot(s.e - det.e, s.n - det.n)
            if dist < bd:
                best, bd = s, dist
        vp = (float(viewpoint[0]), float(viewpoint[1]))
        if best is not None and bd <= cfg.merge_m:
            s = best
            if s.status in ("rejected", "reached", "assigned", "confirmed"):
                return
            displaced = all(math.hypot(vp[0] - v[0], vp[1] - v[1]) >= 0.6 * cfg.relook_offset for v in s.views)
            if displaced:
                s.views.append(vp)
                n = len(s.views)
                s.e, s.n = (s.e * (n - 1) + det.e) / n, (s.n * (n - 1) + det.n) / n
                s.score = max(s.score, det.score)
                if det.snapshot:
                    s.snapshot = det.snapshot
            if len(s.views) >= 2 and s.status == "candidate":
                self._confirm(s, "persistence check")
            return
        s = Survivor(self._next_sid, det.e, det.n, det.score, self.t_mission(), views=[vp], snapshot=det.snapshot)
        self._next_sid += 1
        self.survivors[s.id] = s
        self.event(f"{d.name} candidate #{s.id} (score {det.score:.2f})")
        self.log("candidate", sid=s.id, e=s.e, n=s.n, pos=self.lonlat(s.e, s.n),
                 drone=d.id, score=det.score)
        if d.agent:              # the sim's sensor already confirmed it, and the agent cannot fly a re-look
            self._confirm(s, "sim sensor")
        elif not self.options["relook"]:
            self._confirm(s, "re-look off")
        elif d.state == "SEARCH" and d.relook is None and not d.confirming:
            d.relook = {"sid": s.id, "from": vp}

    def _confirm(self, s: Survivor, how):
        s.status, s.t_confirmed = "confirmed", self.t_mission()
        self.event(f"survivor #{s.id} confirmed ({how})")
        self.log("confirmed", sid=s.id, e=s.e, n=s.n, pos=self.lonlat(s.e, s.n),
                 how=how, true=self._is_true(s))

    def _is_true(self, s):
        if self.truth_en is None or not len(self.truth_en):
            return None
        return bool(np.min(np.hypot(self.truth_en[:, 0] - s.e, self.truth_en[:, 1] - s.n)) <= self.cfg.merge_m)

    # --------------------------------------------------------------- flight
    async def _ensure_airborne(self, d: Drone, alt):
        v = d.vehicle
        if not v.tel.in_air:
            d.task_text = "taking off"
            await v.takeoff(alt)
            d.carrot = None

    async def _follow(self, d: Drone, speed, until_remaining=0.0, check_relook=False):
        """Stream setpoints along d.path. Returns early when the remaining path is short."""
        v, cfg = d.vehicle, self.cfg
        dt = 0.1
        fl = self.flight
        if d.carrot is None:
            d.carrot = self._pos_agl(d)
            d.carrot_v = float(np.hypot(*v.tel.vel[:2]))
        # the operator's speed cap also caps the carrot: a faster carrot leaves the drone behind
        top = speed if self.drone_speed is None else min(speed, self.drone_speed)
        while d.path:
            if check_relook and d.relook is not None:
                return "relook"
            rem = float(np.linalg.norm(d.path[0] - d.carrot)) + sum(
                float(np.linalg.norm(d.path[k + 1] - d.path[k])) for k in range(len(d.path) - 1))
            if rem <= until_remaining:
                return "near_end"
            tgt = d.path[0]
            leg = tgt - d.carrot
            L = float(np.linalg.norm(leg))
            if L < 1e-3:
                d.path.pop(0)
                continue
            u = leg / L
            spd = min(top, self._brake_speed(d, u, L, top), d.carrot_v + cfg.accel * dt)
            # the carrot leads the drone by at most a few m/s along the leg: after a reversal (a re-look
            # behind it) the drone first brakes, instead of chasing a carrot that already runs the other way
            uh = float(np.hypot(u[0], u[1]))
            if uh > 0.3:
                along = float(np.dot(v.tel.vel[:2], u[:2])) / uh
                spd = min(spd, max(along, 0.0) + LEAD_SPEED)
            if abs(u[2]) > 1e-3:
                spd = min(spd, cfg.climb / abs(u[2]))
            spd = max(spd, 0.3)            # never stall short of the end point
            sp = fl.to_vehicle(d.id, v.home, d.carrot, u)
            # the lag guard is about the horizontal tracking: the height of the setpoint
            # carries the terrain term (ground at the target - ground at home), which is
            # tens of metres over the hills and would freeze the carrot for good
            lag = float(np.linalg.norm(sp[:2] - v.tel.pos[:2]))
            blocked = self._sense(d, dt)
            # a vehicle that steers from where it will be (track_lead) trails by that much of its speed
            guard = max(8.0, spd * 1.5) + v.track_lead * float(np.hypot(*v.tel.vel[:2]))
            adv = 0.0 if blocked or lag > guard else spd * dt
            # while the carrot waits for the drone it starts again from the drone's speed, not its own
            d.carrot_v = spd if adv > 0 else min(spd, float(np.hypot(*v.tel.vel[:2])))
            step = min(adv, L)
            d.carrot = d.carrot + u * step
            if step >= L - 1e-6:
                d.path.pop(0)
            # no feed-forward while the carrot waits: a drone that overshot it would hold
            # where the forward push balances the pull back, beyond the lag guard, for good
            await v.setpoint(fl.to_vehicle(d.id, v.home, d.carrot, u), u * spd if adv > 0 else np.zeros(3))
            await self.clock.sleep(dt)
        # settle at the end point
        for _ in range(100):
            sp = fl.to_vehicle(d.id, v.home, d.carrot)
            if np.linalg.norm(v.tel.pos - sp) < 1.5:
                break
            await v.setpoint(sp, np.zeros(3))
            await self.clock.sleep(dt)
        return "done"

    def _brake_speed(self, d: Drone, u, L, top):
        """Highest carrot speed from which the drone can still slow to the turn speed at each corner
        ahead and stop at the end of the path, braking at MissionConfig.accel."""
        cfg = self.cfg
        a, reach = cfg.accel, top * top / (2 * cfg.accel)
        dist, lim, k = L, math.inf, 0
        while True:
            if k + 1 >= len(d.path):
                return min(lim, math.sqrt(2 * a * dist))            # the end of the path: stop
            nxt = d.path[k + 1] - d.path[k]
            nl = float(np.linalg.norm(nxt))
            if nl > 1e-3:
                v_turn = self._turn_speed(u, nxt / nl)
                if v_turn is not None:
                    lim = min(lim, math.sqrt(v_turn * v_turn + 2 * a * dist))
                u = nxt / nl
            dist += nl
            k += 1
            if dist > reach:                                         # corners further on do not limit yet
                return lim

    def _turn_speed(self, u, w):
        """Carrot speed at the joint from direction u to w (unit 3D vectors); None for a straight joint.
        Judged on the horizontal parts: a climb or descent after level flight stops the drone."""
        nu, nw = float(np.hypot(u[0], u[1])), float(np.hypot(w[0], w[1]))
        if nu < 0.3:                       # leaving a climb or descent: the climb rate already keeps it slow
            return None
        if nw < 0.3:                       # level flight into a climb or descent: stop over the point
            return 0.0
        cos = float(np.dot(u[:2], w[:2])) / (nu * nw)
        if cos > 0.995:
            return None
        return self.cfg.turn_speed * (1 + cos)          # right angle: turn_speed, U-turn: 0

    def _pos_agl(self, d: Drone):
        """Vehicle position with the height above the ground below it (the mission's height frame)."""
        p = d.vehicle.tel.pos.copy()
        p[2] = self.flight.agl(d.vehicle.home, p)
        return p

    def _sense(self, d: Drone, dt):
        """Terrain hold and avoidance from the sim sensors; True while the way ahead is blocked."""
        self.flight.record(d.vehicle.home, d.vehicle.tel.pos)
        if self.sensors is None:
            return False
        rng, depth = self.sensors.latest(d.id)
        return self.flight.sense(d.id, dt, rng, depth)

    async def _fly_to(self, d: Drone, e, n, alt, speed=None):
        p = self._pos_agl(d)
        cur_alt = p[2]
        pts = []
        if alt > cur_alt + 0.5:
            pts.append(np.array([p[0], p[1], alt]))
            pts.append(np.array([e, n, alt]))
        else:
            pts.append(np.array([e, n, cur_alt]))
            if alt < cur_alt - 0.5:
                pts.append(np.array([e, n, alt]))
        d.path = pts
        d.carrot = None
        await self._follow(d, speed or self.cfg.transit_speed)

    def _cell_point(self, cell, alt):
        e, n = self.mission.grid.cell_center(*cell)
        return np.array([float(e), float(n), alt])

    def _entry_cell(self, d: Drone):
        g = self.mission.grid
        p = d.vehicle.tel.pos
        rr, cc = np.nonzero(g.mask)
        e, n = g.cell_center(rr, cc)
        cost = np.hypot(e - p[0], n - p[1])
        for o in self.drones:
            if o is d or o.state != "SEARCH":
                continue
            cells = list(o.segment) + [c for c in (o.plan_cell, o.entry) if c is not None]
            if o.plan_cell is not None:
                cells.append(self.drone_cell(o))
            for (r, c) in cells:
                close = np.maximum(np.abs(rr - r), np.abs(cc - c)) <= self.policy.cfg.min_sep + 1
                cost = np.where(close, cost + 1e6, cost)
        k = int(np.argmin(cost))
        return int(rr[k]), int(cc[k])

    # ------------------------------------------------------------- behaviors
    async def _search(self, d: Drone):
        if self.mission.mode == AGENT_MODE and getattr(d.vehicle, "agent_capable", False):
            await self._agent_search(d)
            return
        cfg = self.cfg
        alt_t = cfg.transit_alt[d.id % len(cfg.transit_alt)]
        await self._ensure_airborne(d, alt_t)
        entry = self._entry_cell(d)
        d.entry = entry
        d.task_text = f"transit to cell {entry}"
        ep = self._cell_point(entry, cfg.search_alt)
        await self._fly_to(d, ep[0], ep[1], alt_t)
        await self._fly_to(d, ep[0], ep[1], cfg.search_alt)
        d.plan_cell, d.heading, d.segment = entry, -1, [entry]
        d.task_text = "searching"
        seg_time = self.policy.cfg.ring * self.policy.cfg.t_straight
        while True:
            m = self.mission
            if m.belief.sum() < STOP_FRACTION * m.belief0:
                await self._follow(d, cfg.cruise)
                self.stop_mission("search complete")
                return
            if d.relook is not None:
                await self._relook(d)
                continue
            a, cells, alt = self._decide(d, d.plan_cell)
            if len(cells) == 1:                      # hold: look again and wait one segment
                await self._follow(d, cfg.cruise, check_relook=True)
                d.segment = cells
                self._look(d, d.plan_cell, again=True)
                d.heading = -1
                await self.clock.sleep(seg_time)
                continue
            d.segment = cells
            d.heading = self._heading_of(a)
            d.path.extend(self._cell_point(c, alt) for c in cells)
            d.plan_cell = cells[-1]
            await self._follow(d, cfg.cruise, until_remaining=cfg.cell_m, check_relook=True)

    async def _agent_search(self, d: Drone):
        """The sim's trained agent flies the drone until another task takes it over; any setpoint ends its flight."""
        await self.clock.sleep(0.5)        # a setpoint of the cancelled task still on its way would take the drone back
        await d.vehicle.fly_agent(self.agent_limits, self.agent)
        d.agent, d.heading, d.segment = True, -1, []
        d.task_text = "searching (Unity agent)"
        try:
            while True:
                m = self.mission
                if m.belief.sum() < STOP_FRACTION * m.belief0:
                    self.stop_mission("search complete")
                    return
                if not d.vehicle.agent_flying:       # Unity restarted its mission: the drone is back on its pad
                    await d.vehicle.fly_agent(self.agent_limits, self.agent)
                d.plan_cell = self.drone_cell(d)
                await self.clock.sleep(1.0)
        finally:
            d.agent = False

    async def set_agent_limits(self, limits: str):
        if limits not in ("training", "bridge"):
            raise ValueError("limits must be training or bridge")
        self.agent_limits = limits
        for d in self.drones:
            if d.agent:
                await d.vehicle.fly_agent(limits, self.agent)
        self.event("agent flight limits: " + ("training (speed cap off)" if limits == "training" else "operator"))

    async def set_agent(self, agent: str):
        if self.refresh_agents is not None:
            with contextlib.suppress(Exception):
                await self.refresh_agents()
        labels = {a["id"]: a["label"] for a in self.agents}
        if agent not in labels:
            raise ValueError(f"agent must be one of {list(labels)}")
        self.agent = agent
        for d in self.drones:
            if d.agent:
                await d.vehicle.fly_agent(self.agent_limits, agent)
        self.event(f"Unity agent: {labels[agent]}")

    async def _relook(self, d: Drone):
        """Scripted re-look from a displaced viewpoint (plan 4.7)."""
        cfg = self.cfg
        rl = d.relook
        s = self.survivors.get(rl["sid"])
        if s is None or s.status != "candidate" or not self.options["relook"]:
            d.relook = None
            return
        d.confirming = True
        self.metrics["relooks"] += 1
        saved, saved_text = d.path, d.task_text
        d.task_text = f"re-look #{s.id}"
        fe, fn = rl["from"]
        vec = np.array([s.e - fe, s.n - fn])
        nv = float(np.linalg.norm(vec))
        perp = np.array([-vec[1], vec[0]]) / nv if nv > 1e-3 else np.array([1.0, 0.0])
        pt = np.array([s.e, s.n]) + perp * cfg.relook_offset
        d.path = [np.array([pt[0], pt[1], cfg.search_alt])]
        await self._follow(d, cfg.cruise)
        await d.vehicle.hold()
        await self.clock.sleep(cfg.relook_hold)
        await self._detect(d, d.vehicle.tel.pos.copy(), d.vehicle.tel.yaw, relook=True)
        if s.status == "candidate":
            self.event(f"#{s.id} not seen on re-look; waiting for operator", "warn")
        d.path, d.task_text = saved, saved_text
        d.relook, d.confirming = None, False

    async def _rescue(self, d: Drone, s: Survivor):
        cfg = self.cfg
        alt = cfg.transit_alt[d.id % len(cfg.transit_alt)]
        await self._ensure_airborne(d, alt)
        d.target = np.array([s.e, s.n])
        d.task_text = f"to survivor #{s.id}"
        await self._fly_to(d, s.e, s.n, alt)
        d.state, d.task_text = "ON_SURVIVOR", f"over survivor #{s.id}"
        await self._fly_to(d, s.e, s.n, 10.0)
        await d.vehicle.hold()
        await self.clock.sleep(cfg.rescue_hold)
        s.status, s.t_reached = "reached", self.t_mission()
        if self._is_true(s) is False:
            self.metrics["false_dispatch"] += 1
        self.event(f"{d.name} reached survivor #{s.id}")
        self.log("reached", sid=s.id, drone=d.id, pos=self.lonlat(s.e, s.n), true=self._is_true(s))
        d.target, d.survivor = None, None
        m = self.mission
        if d.resume_search and m and m.status == "running" and not m.stopping:
            self._set_task(d, self._search(d), "SEARCH", "back to search")
            return
        point = d.standby_point if d.standby_point is not None else d.vehicle.home[:2]
        self._set_task(d, self._standby(d, point), "STANDBY", "to staging point")

    async def _standby(self, d: Drone, point):
        cfg = self.cfg
        d.standby_point = np.asarray(point, float)
        if np.hypot(*(d.vehicle.tel.pos[:2] - d.standby_point)) > 2.0 or d.vehicle.tel.in_air:
            alt = cfg.transit_alt[d.id % len(cfg.transit_alt)]
            await self._ensure_airborne(d, alt)
            await self._fly_to(d, d.standby_point[0], d.standby_point[1], alt)
            d.task_text = "landing at staging point"
            await d.vehicle.land()
        d.task_text = "landed at staging point"

    async def _return(self, d: Drone, reason):
        alt = self.cfg.transit_alt[d.id % len(self.cfg.transit_alt)]
        d.task_text = f"return home: {reason}"
        await d.vehicle.return_home(alt)
        d.state, d.task_text = "LANDED", f"landed ({reason})"
        self.log("state", drone=d.id, state="LANDED", text=reason)

    async def _abort(self, d: Drone):
        await d.vehicle.hold()
        await self.clock.sleep(5.0)
        await self._return(d, "lost link")

    # --------------------------------------------------------------- dispatch
    def _eta(self, d: Drone, e, n):
        """Seconds until d could be over (e, n): the rest of its current rescue, then the flight there."""
        cfg, p = self.cfg, d.vehicle.tel.pos
        start, t = p[:2], 0.0
        if d.state in ("RESCUE_TRANSIT", "ON_SURVIVOR") and d.target is not None:
            if d.state == "RESCUE_TRANSIT":
                t += math.hypot(*(d.target - p[:2])) / cfg.transit_speed
            t += cfg.rescue_hold
            start = d.target
        elif d.state not in ("IDLE", "STANDBY", "SEARCH"):
            return math.inf
        elif not d.vehicle.tel.in_air:
            t += TAKEOFF_S
        t += math.hypot(e - start[0], n - start[1]) / cfg.transit_speed
        home = d.vehicle.home
        back = math.hypot(e - home[0], n - home[1]) / cfg.transit_speed
        if d.battery_s < t + cfg.rescue_hold + back + cfg.land_s + cfg.reserve_s:
            return math.inf
        return t

    def _search_loss(self, d: Drone, away):
        """Survivor-seconds of delay when d stops searching for `away` s: the detections it would
        make in that time (belief of its segment, one cell look per cell_m / cruise s), each found `away` s later."""
        m, cfg = self.mission, self.cfg
        cells = [tuple(c) for c in d.segment] or [self.drone_cell(d)]
        b = float(np.mean([m.belief[c] if m.grid.in_grid(*c) and m.grid.mask[c] else 0.0 for c in cells]))
        return cfg.pd * b * cfg.cruise / cfg.cell_m * away * away

    def _auto_switch(self):
        """Send a search drone to a confirmed survivor when that beats waiting for a rescue drone.

        For each survivor (oldest first) the wait is the soonest a rescue drone can be over it,
        after the survivors already planned for it. A search drone switches when its arrival plus
        the search delay it causes (_search_loss) beats the wait by MissionConfig.switch_margin_s;
        it searches again after the rescue. Not in AGENT_MODE: the Unity agent has its own rules.
        """
        m, cfg = self.mission, self.cfg
        if m.mode == AGENT_MODE:
            return
        conf = sorted((s for s in self.survivors.values() if s.status == "confirmed"),
                      key=lambda s: s.t_confirmed or 0.0)
        if not conf:
            return
        rescuers = [d for d in self.drones if d.role == "rescue"
                    and d.state in ("IDLE", "STANDBY", "RESCUE_TRANSIT", "ON_SURVIVOR")]
        planned = {d.id: 0.0 for d in rescuers}       # s of planned rescues ahead of the next one
        searchers = [d for d in self.drones if d.state == "SEARCH" and not d.agent and d.plan_cell is not None
                     and d.relook is None and not d.confirming]
        for s in conf:
            wait, r = min(((self._eta(d, s.e, s.n) + planned[d.id], d) for d in rescuers),
                          key=lambda x: x[0], default=(math.inf, None))
            best = None
            for d in searchers:
                t = self._eta(d, s.e, s.n)
                if math.isinf(t):
                    continue
                rep = self._replacement(d)
                p = d.vehicle.tel.pos
                # the search gap: until a replacement arrives, or until d is back (out, hold, back)
                gap = self._eta(rep, p[0], p[1]) if rep is not None else 2 * t + cfg.rescue_hold
                cost = t + self._search_loss(d, min(gap, 2 * t + cfg.rescue_hold))
                if best is None or cost < best[0]:
                    best = (cost, d, t)
            if best is not None and best[0] + cfg.switch_margin_s < wait:
                cost, d, t = best
                searchers.remove(d)
                self.log("switch", sid=s.id, drone=d.id, eta=round(t, 1), cost=round(cost, 1),
                         wait=None if math.isinf(wait) else round(wait, 1))
                self.assign(d, s, f"auto switch: {t:.0f} s vs {wait:.0f} s wait")
                d.resume_search = True
            elif r is not None and not math.isinf(wait):
                planned[r.id] = wait + cfg.rescue_hold

    def _dispatch(self):
        conf = [s for s in self.survivors.values() if s.status == "confirmed"]
        if not conf:
            return
        avail = [d for d in self.drones if d.role == "rescue" and d.state in ("IDLE", "STANDBY")]
        if not avail:
            return
        cost = np.zeros((len(avail), len(conf)))
        for i, d in enumerate(avail):
            p = d.vehicle.tel.pos
            for j, s in enumerate(conf):
                t = math.hypot(s.e - p[0], s.n - p[1]) / self.cfg.transit_speed
                if not d.vehicle.tel.in_air:
                    t += 10.0
                cost[i, j] = t
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            self.assign(avail[i], conf[j], "auto")

    def assign(self, d: Drone, s: Survivor, how):
        if s.status in ("reached", "rejected"):
            raise ValueError(f"survivor #{s.id} is {s.status}")
        for o in self.drones:                    # one drone per survivor
            if o.survivor == s.id and o is not d:
                o.survivor = None
                self._set_task(o, self._standby(o, o.standby_point if o.standby_point is not None else o.vehicle.home[:2]),
                               "STANDBY", "to staging point")
        if s.status == "candidate":
            self._confirm(s, "operator dispatch")
        s.status, s.assigned = "assigned", d.id
        d.survivor = s.id
        self._set_task(d, self._rescue(d, s), "RESCUE_TRANSIT", f"to survivor #{s.id}")
        self.event(f"{d.name} dispatched to #{s.id} ({how})")
        if d.role == "search":
            self._replace(d)

    def _replacement(self, leaving: Drone):
        if not self.options["auto_replace"] or not self.mission or self.mission.stopping:
            return None
        for d in self.drones:
            if d is not leaving and d.role == "search" and d.state in ("STANDBY", "IDLE") and d.battery_s > 300:
                return d
        return None

    def _replace(self, leaving: Drone):
        d = self._replacement(leaving)
        if d is not None:
            self._set_task(d, self._search(d), "SEARCH", "replacement")
            self.event(f"{d.name} launched to replace {leaving.name}")

    def _auto_stage(self):
        m = self.mission
        if not self.options["auto_stage"] or self.now() - self._last_stage < self.cfg.stage_period:
            return
        self._last_stage = self.now()
        b = np.where(m.grid.mask, m.belief, 0.0)
        if b.sum() <= 0:
            return
        rr, cc = np.meshgrid(np.arange(m.grid.g), np.arange(m.grid.g), indexing="ij")
        r, c = (b * rr).sum() / b.sum(), (b * cc).sum() / b.sum()
        e, n = m.grid.cell_center(r, c)
        for d in self.drones:
            if d.role == "rescue" and d.state == "STANDBY":
                old = d.standby_point if d.standby_point is not None else d.vehicle.home[:2]
                if math.hypot(e - old[0], n - old[1]) > 3 * self.cfg.cell_m:
                    self._set_task(d, self._standby(d, (float(e), float(n))), "STANDBY", "auto-stage")

    # ------------------------------------------------------------------ tick
    async def _loop(self):
        cfg = self.cfg
        t_last = self.now()
        k = 0
        while True:
            await self.clock.sleep(cfg.tick)
            t = self.now()
            dt = t - t_last
            t_last = t
            k += 1
            try:
                self._tick(t, dt)
            except Exception as exc:
                self.event(f"tick error: {exc!r}", "error")
            if k % max(1, int(0.2 / cfg.tick)) == 0:
                self._publish(self.state_msg())
            if self.mission and k % max(1, int(1.0 / cfg.tick)) == 0:
                self._publish(self.belief_msg())
            # Replay rows (plan 11.2): tracks at 1 Hz, belief PNGs at 0.2 Hz.
            if self.mission and self._log:
                if dt > 0 and (t - self._last_track) >= 1.0:
                    self._last_track = t
                    self.log_track()
                if dt > 0 and (t - self._last_belief_log) >= 5.0:
                    self._last_belief_log = t
                    self.log_belief()

    def _tick(self, t, dt):
        cfg = self.cfg
        m = self.mission
        for d in self.drones:
            tel = d.vehicle.tel
            if tel.in_air:
                d.battery_s -= dt
            # lost link
            # a long gap between ticks means this process stalled, not the link
            if dt > cfg.link_timeout / 2:
                # telemetry polls wait for the loop too: give them one timeout to report again
                self._link_grace = t + cfg.link_timeout
            if t >= self._link_grace and d.state in AIRBORNE and d.state != "ABORT" and t - tel.t_update > cfg.link_timeout:
                self.event(f"{d.name} lost link", "error")
                was_search = d.state == "SEARCH"
                self._set_task(d, self._abort(d), "ABORT", "lost link: hover then RTL")
                if was_search:
                    self._replace(d)
                continue
            # battery rule
            if d.state in ("SEARCH", "RESCUE_TRANSIT", "ON_SURVIVOR") or (d.state == "STANDBY" and tel.in_air):
                dist = math.hypot(tel.pos[0] - d.vehicle.home[0], tel.pos[1] - d.vehicle.home[1])
                need = dist / cfg.transit_speed + cfg.land_s + cfg.reserve_s
                if d.battery_s <= need:
                    self.event(f"{d.name} battery rule: returning home", "warn")
                    was_search = d.state == "SEARCH"
                    if d.survivor is not None and d.survivor in self.survivors:
                        s = self.survivors[d.survivor]
                        if s.status == "assigned":
                            s.status, s.assigned = "confirmed", None
                    d.survivor = None
                    self._set_task(d, self._return(d, "battery"), "RETURNING", "battery")
                    if was_search:
                        self._replace(d)
                    continue
            # looks from the actual position; transit flights (rescue, return, staging) look too, as a search look
            searching = d.state == "SEARCH" and d.plan_cell is not None
            transit = d.state in TRANSIT_LOOKS and tel.in_air
            high = (d.vehicle.agl or 0.0) > AGENT_LOOK_AGL if d.agent else self._pos_agl(d)[2] > cfg.search_alt - 3
            if m and (searching or transit) and high:
                cell = self.drone_cell(d)
                if cell != d.last_cell:
                    d.last_cell = cell
                    self._look(d, cell)
            else:
                d.view, d.last_cell = set(), None     # out of view: the next look is a new pass
        if m and m.status == "running":
            if self.options["auto_switch"] and not m.stopping:
                self._auto_switch()
            if self.options["auto_dispatch"]:
                self._dispatch()
            self._auto_stage()
            busy = any(d.state in AIRBORNE for d in self.drones)
            if not busy and not m.stopping and self.t_mission() > 30.0:
                m.stopping = True
                self.event("no drone is searching", "warn")
                self.log("no_search", states=[(d.name, d.state, d.task_text, bool(d.vehicle.tel.in_air)) for d in self.drones])
            if m.stopping and not busy:
                m.status = "finished"
                self.event("mission finished")
                self.log("finished", found=self.expected_found(), t=self.t_mission())

    # --------------------------------------------------------------- commands
    async def command(self, msg: dict):
        kind = msg.get("type")
        d = self.drones[int(msg["drone"])] if "drone" in msg else None
        s = self.survivors[int(msg["survivor"])] if msg.get("survivor") is not None else None
        if kind == "start_mission":
            g = self._grid_size(msg["polygon"])
            if g not in self._warm_grids:
                # shown on the UI only: a log row before the mission row would split the replay
                self._publish({"type": "event", "t": 0.0, "level": "info",
                               "text": "preparing the search policies"})
            await asyncio.to_thread(self.warm_policies, g)
            self.start_mission(polygon=msg["polygon"], lkp=msg.get("lkp"), mode=msg.get("mode", "rl"),
                               n_search=int(msg.get("n_search", 2)),
                               expected_survivors=float(msg.get("expected_survivors", 3.0)),
                               options=msg.get("options"), scenario=msg.get("scenario", "wilderness"),
                               lkp_time_h=msg.get("lkp_time_h"), regions=msg.get("regions"),
                               water_level_m=float(msg.get("water_level_m") or 0.0))
            self._publish(self.prior_msg())
        elif kind == "stop_mission":
            self.stop_mission()
        elif kind == "continue_mission":
            self.continue_mission()
        elif kind == "restart_mission":
            await self.restart_mission()
            self._publish(self.prior_msg())
        elif kind == "reset_mission":
            await self.reset_mission()
        elif kind == "new_survivors":
            await self.new_survivors(msg.get("count"), msg.get("mix"))
        elif kind == "recharge":
            self.recharge(d)
        elif kind == "set_graphics":
            await self.set_graphics(msg["preset"])
        elif kind == "set_drone_speed":
            await self.set_drone_speed(msg["speed"])
        elif kind == "set_agent_limits":
            await self.set_agent_limits(msg["limits"])
        elif kind == "set_agent":
            await self.set_agent(msg["agent"])
        elif kind == "set_mode":
            if msg["mode"] not in self.modes:
                raise ValueError(f"mode must be one of {self.modes}")
            if msg["mode"] in MODES_V2:    # compiled at mission start; this is a no-op then
                get_policy_v2(self.mission.grid.g).warmup(msg["mode"])
            old, self.mission.mode = self.mission.mode, msg["mode"]
            self.event(f"search mode: {msg['mode']}")
            if (old == AGENT_MODE) != (msg["mode"] == AGENT_MODE):
                # the agent takes over a searching drone, or hands it back to the search mode
                for o in self.drones:
                    if o.state == "SEARCH" and getattr(o.vehicle, "agent_capable", False):
                        self._set_task(o, self._search(o), "SEARCH", "search mode changed")
        elif kind == "recall":
            was_search = d.state == "SEARCH"
            self._release_survivor(d)
            self._set_task(d, self._return(d, "recall"), "RETURNING", "recall")
            self.event(f"{d.name} recalled")
            if was_search:
                self._replace(d)
        elif kind == "standby":
            point = self.en(msg["point"]) if msg.get("point") else d.vehicle.home[:2].copy()
            was_search = d.state == "SEARCH"
            self._release_survivor(d)
            self._set_task(d, self._standby(d, point), "STANDBY", "to staging point")
            if was_search:
                self._replace(d)
        elif kind == "launch":
            if not self.mission or self.mission.status != "running":
                raise ValueError("no running mission")
            if msg.get("task", "search") == "search":
                self._release_survivor(d)
                self._set_task(d, self._search(d), "SEARCH", "launched")
            else:
                if s is None:
                    conf = [x for x in self.survivors.values() if x.status == "confirmed"]
                    if not conf:
                        raise ValueError("no confirmed survivor to rescue")
                    p = d.vehicle.tel.pos
                    s = min(conf, key=lambda x: math.hypot(x.e - p[0], x.n - p[1]))
                self.assign(d, s, "operator")
        elif kind == "dispatch":
            self.assign(d, s, "operator")
        elif kind == "confirm":
            if s.status == "candidate":
                self._confirm(s, "operator")
        elif kind == "reject":
            if s.status in ("candidate", "confirmed"):
                s.status = "rejected"
                self.event(f"survivor #{s.id} rejected by operator")
        elif kind == "set_options":
            self.options.update({k: bool(v) for k, v in msg.items() if k in self.options})
        elif kind == "set_speed":
            self.clock.set_speed(float(msg["speed"]))
        elif kind == "select":
            self.select(msg.get("drone"), msg.get("survivor"))
        else:
            raise ValueError(f"unknown command {kind}")

    def _release_survivor(self, d: Drone):
        if d.survivor is not None and d.survivor in self.survivors:
            s = self.survivors[d.survivor]
            if s.status == "assigned":
                s.status, s.assigned = "confirmed", None
        d.survivor, d.target = None, None

    # ---------------------------------------------------------------- outputs
    def expected_found(self):
        m = self.mission
        return float(m.belief0 - m.belief.sum()) if m else 0.0

    def select(self, drone=None, survivor=None):
        """Operator selection for the video panels (plan 11.1)."""
        if drone is not None:
            self.selected = int(drone)
        elif survivor is not None and int(survivor) in self.survivors:
            s = self.survivors[int(survivor)]
            # point the follow view at the nearest drone to the survivor
            k = min(range(len(self.drones)), key=lambda i: float(np.hypot(
                self.drones[i].vehicle.tel.pos[0] - s.e, self.drones[i].vehicle.tel.pos[1] - s.n)))
            self.selected = self.drones[k].id
        return getattr(self, "selected", 0)

    def latest_detections(self, drone):
        last = getattr(self.detector, "last", None)
        return None if last is None else last.get(int(drone))

    def state_msg(self, truth=False):
        m = self.mission
        drones = []
        for d in self.drones:
            tel = d.vehicle.tel
            seg = []
            if m and d.segment and d.state == "SEARCH":
                seg = [self.lonlat(*m.grid.cell_center(*c)) for c in d.segment]
            drones.append({
                "id": d.id, "name": d.name, "role": d.role, "state": d.state,
                "pos": self.lonlat(tel.pos[0], tel.pos[1]), "alt": round(float(tel.pos[2]), 1),
                "heading": round(float(tel.yaw), 1), "battery_s": round(max(d.battery_s, 0.0), 1),
                "battery_frac": round(max(d.battery_s, 0.0) / self.cfg.battery_s, 3),
                "segment": seg,
                "target": self.lonlat(*d.target) if d.target is not None else None,
                "standby_point": self.lonlat(*d.standby_point) if d.standby_point is not None else None,
                "task": d.task_text,
                "level": getattr(d, "level", None) or ("transit" if d.state in ("RESCUE_TRANSIT", "RETURNING")
                                                       else "low" if d.state == "SEARCH" else None),
                "alt_agl": round(float(self._pos_agl(d)[2]), 1),
                "sortie": getattr(d, "sortie", 1),
                "swap_remaining": getattr(d, "swap_remaining", None),
                # camera ground footprint half sizes; the sim detector sees one cell
                "footprint_m": d.vehicle.footprint_m() or [self.cfg.cell_m / 2, self.cfg.cell_m / 2],
            })
        surv = [{
            "id": s.id, "pos": self.lonlat(s.e, s.n), "status": s.status, "score": round(s.score, 3),
            "views": len(s.views), "has_snapshot": s.snapshot is not None, "assigned": s.assigned,
            "t_detected": round(s.t_detected, 1),
            "t_confirmed": None if s.t_confirmed is None else round(s.t_confirmed, 1),
            "t_reached": None if s.t_reached is None else round(s.t_reached, 1),
        } for s in self.survivors.values()]
        mission = {"status": "idle", "mode": None, "polygon": None, "lkp": None, "expected_total": 0.0,
                   "remaining_fraction": 1.0, "expected_found": 0.0, "covered_fraction": 0.0}
        if m:
            inside = m.grid.mask
            mission = {
                "status": m.status, "stopping": m.stopping, "mode": m.mode, "polygon": m.polygon, "lkp": m.lkp,
                "expected_total": m.expected,
                "remaining_fraction": round(float(m.belief.sum() / max(m.belief0, 1e-9)), 4),
                "expected_found": round(self.expected_found(), 3),
                "covered_fraction": round(float(((m.looks > 0) & inside).sum() / inside.sum()), 4),
            }
        tr = None
        if truth and self.truth_en is not None:
            tr = [self.lonlat(e, n) for e, n in self.truth_en]
        return {"type": "state", "t": round(self.t_mission(), 2), "speed": self.clock.speed,
                "drone_speed": self.drone_speed, "agent_limits": self.agent_limits, "agent": self.agent,
                "graphics": self.graphics,
                "mission": mission, "drones": drones, "survivors": surv, "options": dict(self.options),
                "truth": tr}

    def _blend(self, grid, layers, regions=None):
        """Mission prior from the prior components plus operator regions (plan 6.5), sums to 1 in the mask."""
        mask = grid.mask
        base = layers["blend"] if "blend" in layers else sum(layers.values())
        base = np.where(mask, np.asarray(base, np.float64), 0.0)
        base = base / max(base.sum(), 1e-12)
        if regions:
            ex, ny = grid.cell_center(*np.meshgrid(np.arange(grid.g), np.arange(grid.g), indexing="ij"))
            op, wsum = np.zeros_like(base), 0.0
            for reg in regions:
                poly = np.asarray(reg["polygon"], float)
                e, n = self.frame.to_en(poly[:, 0], poly[:, 1])
                inside = points_in_polygon(ex, ny, np.stack([e, n], -1)) & mask
                if inside.any():
                    w = float(np.clip(reg.get("weight", 0.3), 0.0, 1.0))
                    op += w * inside / inside.sum()
                    wsum += w
            if wsum > 0:
                wsum = min(wsum, 0.9)
                base = (1 - wsum) * base + wsum * op / op.sum()
        return (base / base.sum()).astype(np.float32)

    @staticmethod
    def _heat_png(a, mask):
        from PIL import Image
        b = np.where(mask, a, 0.0)
        bmax = max(float(b.max()), 1e-12)
        v = np.clip(1.0 + np.log10(b / bmax + 1e-9) / 3.0, 0, 1)   # three decades below the peak
        rgba = np.zeros(mask.shape + (4,), np.uint8)
        rgba[..., 0] = (40 + 215 * v).astype(np.uint8)
        rgba[..., 1] = (20 + 200 * v ** 1.5).astype(np.uint8)
        rgba[..., 2] = (120 * (1 - v)).astype(np.uint8)
        rgba[..., 3] = np.where(mask & (b > 0), (60 + 150 * v).astype(np.uint8), 0)
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def prior_msg(self):
        """The `prior` message (PROTOCOL 3.3): the blend and each component as heatmap PNGs."""
        m = self.mission
        mask = m.grid.mask
        labels = {"koester": "Koester", "walker": "Walker", "uniform": "Uniform", "flood": "Flood",
                  "regions": "Regions", "roofs": "Roofs", "high_ground": "High ground", "water": "Water"}
        layers = [{"key": k, "label": labels.get(k, k.title()), "png": self._heat_png(np.asarray(v), mask)}
                  for k, v in getattr(m, "prior_layers", {}).items() if k != "blend"]
        return {"type": "prior", "corners": m.grid.corners_lonlat(), "nx": m.grid.g, "ny": m.grid.g,
                "cell_m": m.grid.cell, "scenario": getattr(m, "scenario", "wilderness"),
                "blend": {"key": "blend", "label": "Blend", "png": self._heat_png(m.belief, mask)},
                "layers": layers}

    def belief_msg(self):
        from PIL import Image
        m = self.mission
        mask = m.grid.mask
        b = np.where(mask, m.belief, 0.0)
        bmax = max(float(b.max()), 1e-12)
        v = np.clip(1.0 + np.log10(b / bmax + 1e-9) / 3.0, 0, 1)   # three decades below the peak
        rgba = np.zeros(mask.shape + (4,), np.uint8)
        # dark blue (low) to yellow (high)
        rgba[..., 0] = (40 + 215 * v).astype(np.uint8)
        rgba[..., 1] = (20 + 200 * v ** 1.5).astype(np.uint8)
        rgba[..., 2] = (120 * (1 - v)).astype(np.uint8)
        rgba[..., 3] = np.where(mask, (60 + 150 * v).astype(np.uint8), 0)
        cov = np.zeros(mask.shape + (4,), np.uint8)
        cov[..., :3] = 255
        cov[..., 3] = np.where(mask & (m.looks > 0), 90, 0).astype(np.uint8)

        def png(a):
            buf = io.BytesIO()
            Image.fromarray(a, "RGBA").save(buf, "PNG")
            return base64.b64encode(buf.getvalue()).decode()

        return {"type": "belief", "corners": m.grid.corners_lonlat(), "png": png(rgba), "coverage_png": png(cov)}
