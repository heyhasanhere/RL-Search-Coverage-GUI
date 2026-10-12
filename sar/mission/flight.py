"""Flight layer (plan section 9): altitude floor, terrain hold and obstacle avoidance.

The orchestrator plans in heights above the ground (AGL); the flight layer turns a planned point into
the height each vehicle must fly in its own home-relative frame, and nudges it with the sensors:

  target AGL = max(nominal AGL, surface + clearance) over the next lookahead meters, capped at the ceiling
  surface    = max(canopy height, building height) above the DEM ground
  vehicle u  = target AGL + ground(e, n) - ground(home)

Terrain hold: the downward range sees the canopy top, so it only enforces a minimum clearance above the
measured surface. Avoidance: when the forward depth image shows an obstacle closer than `avoid_m`, the
drone stops and climbs until the way is clear, then rejoins the path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Terrain:
    """DEM, canopy and building heights on the mission grid (row 0 = north edge).

    `origin_en` is the mission-frame (e, n) of the grid's south-west corner.
    """

    def __init__(self, dem, canopy=None, bheight=None, cell_m=30.0, origin_en=(0.0, 0.0)):
        self.dem = np.asarray(dem, np.float32)
        z = np.zeros_like(self.dem)
        self.canopy = z if canopy is None else np.asarray(canopy, np.float32)
        self.bheight = z if bheight is None else np.asarray(bheight, np.float32)
        self.surface_h = np.maximum(self.canopy, self.bheight)
        self.cell = float(cell_m)
        self.e0, self.n0 = map(float, origin_en)
        self.g = self.dem.shape[0]

    @classmethod
    def flat(cls, g=8, cell_m=30.0, origin_en=(0.0, 0.0)):
        return cls(np.zeros((g, g)), cell_m=cell_m, origin_en=origin_en)

    def _frac(self, e, n):
        fx = np.clip((np.asarray(e, float) - self.e0) / self.cell - 0.5, 0, self.g - 1.001)
        fy = np.clip(self.g - (np.asarray(n, float) - self.n0) / self.cell - 0.5, 0, self.g - 1.001)
        return fx, fy

    def ground(self, e, n):
        """Bilinear DEM height."""
        fx, fy = self._frac(e, n)
        c0, r0 = fx.astype(int), fy.astype(int)
        tx, ty = fx - c0, fy - r0
        d = self.dem
        return ((1 - tx) * (1 - ty) * d[r0, c0] + tx * (1 - ty) * d[r0, c0 + 1]
                + (1 - tx) * ty * d[r0 + 1, c0] + tx * ty * d[r0 + 1, c0 + 1])

    def surface(self, e, n):
        """Height of the canopy or roof top above the ground, from the cell it falls in."""
        fx, fy = self._frac(e, n)
        return self.surface_h[np.round(fy).astype(int), np.round(fx).astype(int)]


@dataclass
class FlightConfig:
    clearance: float = 10.0      # m above canopy or roof, from the map
    ceiling: float = 120.0       # m AGL
    min_range: float = 10.0      # m above the measured surface (rangefinder)
    lookahead: float = 60.0      # m: climb early so the floor is never broken along the segment
    avoid_m: float = 15.0        # m: obstacle distance that stops the drone
    climb_step: float = 4.0      # m added per avoidance or terrain-hold correction
    bias_decay: float = 0.5      # m/s: sensor bias relaxes when clear


class FlightLayer:
    def __init__(self, terrain: Terrain | None = None, cfg: FlightConfig | None = None):
        self.t = terrain
        self.cfg = cfg or FlightConfig()
        self.bias: dict[int, float] = {}
        self.blocked: dict[int, bool] = {}
        self.stats = {"floor_violations": 0, "min_clearance": float("inf"), "detours": 0, "detour_s": 0.0}

    def floor_agl(self, e, n):
        if self.t is None:
            return 0.0
        return float(np.max(self.t.surface(e, n))) + self.cfg.clearance

    def target_agl(self, e, n, nominal, heading=None):
        """AGL to fly at (e, n): the nominal height or the floor ahead, whichever is higher."""
        if self.t is None:
            return nominal
        es, ns = np.array([e], float), np.array([n], float)
        if heading is not None and np.linalg.norm(heading[:2]) > 1e-6:
            u = np.asarray(heading[:2], float) / np.linalg.norm(heading[:2])
            s = np.linspace(-self.t.cell / 2, self.cfg.lookahead, 9)
            es, ns = e + u[0] * s, n + u[1] * s
        floor = float(np.max(self.t.surface(es, ns))) + self.cfg.clearance
        return float(min(max(nominal, floor), self.cfg.ceiling))

    def segment_floor(self, p0, p1, step=10.0):
        """Highest floor AGL along a straight segment, for planning checks."""
        if self.t is None:
            return 0.0
        L = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        s = np.linspace(0, 1, max(int(L / step), 1) + 1)
        e, n = p0[0] + s * (p1[0] - p0[0]), p0[1] + s * (p1[1] - p0[1])
        return float(np.max(self.t.surface(e, n))) + self.cfg.clearance

    def to_vehicle(self, vid, home, p_agl, heading=None):
        """Mission point (e, n, nominal AGL) -> vehicle setpoint (e, n, height above home)."""
        e, n, h = float(p_agl[0]), float(p_agl[1]), float(p_agl[2])
        if self.t is None:
            return np.array([e, n, h + self.bias.get(vid, 0.0)])
        agl = min(self.target_agl(e, n, h, heading) + self.bias.get(vid, 0.0), self.cfg.ceiling)
        g = float(self.t.ground(np.array([e]), np.array([n]))[0])
        g0 = float(self.t.ground(np.array([home[0]]), np.array([home[1]]))[0])
        return np.array([e, n, agl + g - g0])

    def agl(self, home, pos):
        """Vehicle position (home-relative height) -> height above the ground below it."""
        if self.t is None:
            return float(pos[2])
        g = float(self.t.ground(np.array([pos[0]]), np.array([pos[1]]))[0])
        g0 = float(self.t.ground(np.array([home[0]]), np.array([home[1]]))[0])
        return float(pos[2]) + g0 - g

    def record(self, home, pos):
        """Clearance bookkeeping for the flight-layer metrics (plan 13.2)."""
        if self.t is None or pos[2] < 5.0:
            return
        a = self.agl(home, pos)
        clear = a - float(self.t.surface(np.array([pos[0]]), np.array([pos[1]]))[0])
        self.stats["min_clearance"] = min(self.stats["min_clearance"], clear)
        if clear < self.cfg.clearance - 3.0:
            self.stats["floor_violations"] += 1

    def sense(self, vid, dt, range_m=None, depth=None):
        """Update the sensor bias for one vehicle. Returns True while the way ahead is blocked."""
        b = self.bias.get(vid, 0.0)
        bumped = False
        if range_m is not None and range_m < self.cfg.min_range:
            b += min(self.cfg.climb_step, self.cfg.min_range - range_m + 1.0)
            bumped = True
        blocked = False
        if depth is not None and np.size(depth):
            d = np.asarray(depth, float)
            h, w = d.shape
            core = d[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
            core = core[np.isfinite(core) & (core > 0.3)]
            if core.size and float(np.min(core)) < self.cfg.avoid_m:
                blocked = True
                b += self.cfg.climb_step * dt * 2
                bumped = True
        if blocked and not self.blocked.get(vid, False):
            self.stats["detours"] += 1
        if blocked:
            self.stats["detour_s"] += dt
        if not bumped:
            b = max(b - self.cfg.bias_decay * dt, 0.0)
        self.bias[vid] = b
        self.blocked[vid] = blocked
        return blocked


class BridgeSensors:
    """Polls `GET /range/<i>` and `GET /depth/<i>` from the sim bridge (plan 9.2) for the flight layer.

    `latest(vid)` never blocks; values older than `max_age` seconds are dropped so a stalled bridge
    cannot keep a stale obstacle in front of a drone.
    """

    def __init__(self, url: str, ids, hz: float = 5.0, max_age: float = 2.0):
        self.url, self.ids, self.period, self.max_age = url.rstrip("/"), list(ids), 1.0 / hz, max_age
        self.data: dict[int, tuple] = {}
        self._task = None

    def start(self):
        import asyncio
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        import asyncio
        import time

        import httpx
        async with httpx.AsyncClient(timeout=2.0) as cl:
            while True:
                for i in self.ids:
                    try:
                        r = (await cl.get(f"{self.url}/range/{i}")).json().get("range")
                        dj = (await cl.get(f"{self.url}/depth/{i}")).json()
                        dep = np.asarray(dj["depth"], np.float32) if "depth" in dj else None
                        self.data[i] = (r, dep, time.monotonic())
                    except (httpx.HTTPError, ValueError, KeyError):
                        pass
                await asyncio.sleep(self.period)

    def latest(self, vid):
        import time
        v = self.data.get(vid)
        if v is None or time.monotonic() - v[2] > self.max_age:
            return None, None
        return v[0], v[1]
