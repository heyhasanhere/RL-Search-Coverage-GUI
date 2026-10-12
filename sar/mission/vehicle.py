"""Vehicle interface, simulation clock, and a kinematic vehicle for fast runs.

All positions are local (east, north, up) meters in the mission frame; up is
above the launch site ground. A vehicle accepts offboard position+velocity
setpoints and simple commands (takeoff, land, return). The kinematic vehicle
moves toward the setpoint with speed and acceleration limits, which is enough
for orchestrator tests, UI development, and Tier 0-style checks without PX4.
"""

import asyncio
import math
import time
from dataclasses import dataclass, field

import numpy as np


class SimClock:
    """Mission time = wall time x speed factor. Speed can change at run time."""

    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self._t0 = 0.0
        self._w0 = time.monotonic()

    def now(self) -> float:
        return self._t0 + (time.monotonic() - self._w0) * self.speed

    def set_speed(self, speed: float):
        self._t0, self._w0, self.speed = self.now(), time.monotonic(), max(speed, 1e-3)

    async def sleep(self, dt: float):
        await asyncio.sleep(max(dt, 0.0) / self.speed)


class BridgeClock(SimClock):
    """Mission time = sim time reported by the bridge (`GET /time`), for sims slower than real time.

    A background thread polls the bridge; between polls the time is extrapolated with the measured
    sim rate, so now() never jumps backwards. set_speed has no effect: the sim sets the pace.
    """

    def __init__(self, url: str, poll: float = 0.1):
        super().__init__(1.0)
        import threading
        import urllib.request
        self._url, self._poll = url.rstrip("/") + "/time", poll
        self._open = urllib.request.urlopen
        self._sim, self._wall, self._rate = self._fetch(), time.monotonic(), 1.0
        self._t0 = self._sim
        self._last = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _fetch(self) -> float:
        import json
        with self._open(self._url, timeout=2.0) as r:
            return float(json.loads(r.read())["sim_time"])

    def _run(self):
        while True:
            time.sleep(self._poll)
            try:
                s, w = self._fetch(), time.monotonic()
            except OSError:
                continue
            if w > self._wall and s >= self._sim:
                self._rate = 0.7 * self._rate + 0.3 * (s - self._sim) / (w - self._wall)
            self._sim, self._wall = s, w

    @property
    def speed(self):
        return getattr(self, "_rate", 1.0)

    @speed.setter
    def speed(self, v):
        pass

    def now(self) -> float:
        t = self._sim - self._t0 + min(time.monotonic() - self._wall, 1.0) * self._rate
        self._last = max(self._last, t)
        return self._last

    def set_speed(self, speed: float):
        pass

    async def sleep(self, dt: float):
        end = self.now() + max(dt, 0.0)
        while (left := end - self.now()) > 0:
            await asyncio.sleep(min(left / max(self._rate, 0.05), 0.05))


@dataclass
class Telemetry:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))   # e, n, u
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0          # degrees, 0 = north, clockwise
    armed: bool = False
    in_air: bool = False
    t_update: float = 0.0     # mission time of the last telemetry message


class Vehicle:
    """Base class. Subclasses implement the async commands."""

    def __init__(self, vid: int, home_en, clock: SimClock):
        self.id = vid
        self.home = np.array([home_en[0], home_en[1], 0.0])
        self.clock = clock
        self.tel = Telemetry(pos=self.home.copy())
        self.tel.t_update = clock.now()

    async def start(self):
        pass

    async def stop(self):
        pass

    async def takeoff(self, alt: float):
        raise NotImplementedError

    async def setpoint(self, pos, vel, yaw_deg=None):
        raise NotImplementedError

    async def land(self):
        raise NotImplementedError

    async def return_home(self, alt: float):
        """Fly home at alt and land (PX4 RTL on real autopilots)."""
        raise NotImplementedError

    async def hold(self):
        await self.setpoint(self.tel.pos.copy(), np.zeros(3))

    async def set_max_speed(self, v: float):
        """Operator speed cap, m/s. Vehicles without a speed limit ignore it."""
        if hasattr(self, "max_speed"):
            self.max_speed = float(v)

    def footprint_m(self):
        """Camera ground footprint (along-track, cross-track) half sizes in meters, or None if unknown."""
        return None


class KinematicVehicle(Vehicle):
    """Point-mass vehicle integrated by its own task at 20 Hz of mission time."""

    def __init__(self, vid, home_en, clock, max_speed=12.0, max_acc=4.0, climb=3.0):
        super().__init__(vid, home_en, clock)
        self.max_speed, self.max_acc, self.climb = max_speed, max_acc, climb
        self._sp_pos = self.home.copy()
        self._sp_vel = np.zeros(3)
        self._mode = "landed"      # landed, offboard, takeoff, landing, rtl
        self._rtl_alt = 30.0
        self._task = None
        self.link_ok = True        # tests set False to simulate a lost link

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()

    async def _run(self):
        dt = 0.05
        t_last = self.clock.now()
        while True:
            await self.clock.sleep(dt)
            t = self.clock.now()
            self._integrate(min(t - t_last, 0.5))
            t_last = t
            if self.link_ok:
                self.tel.t_update = t

    def _integrate(self, dt):
        p, v = self.tel.pos, self.tel.vel
        if self._mode == "landed":
            self.tel.vel = np.zeros(3)
            return
        if self._mode == "rtl":
            horiz = np.hypot(*(self.home[:2] - p[:2]))
            if p[2] < self._rtl_alt - 0.5 and horiz > 2.0:
                tgt = np.array([p[0], p[1], self._rtl_alt])
            elif horiz > 1.0:
                tgt = np.array([self.home[0], self.home[1], max(p[2], self._rtl_alt)])
            else:
                self._mode = "landing"
                tgt = p
            self._sp_pos, self._sp_vel = tgt, np.zeros(3)
        if self._mode == "landing":
            self._sp_pos = np.array([p[0], p[1], p[2]])
            self._sp_vel = np.array([0.0, 0.0, -1.5])
        # P controller on position plus velocity feed-forward, then limits
        err = self._sp_pos - p
        want = self._sp_vel + 1.0 * err
        hs = np.hypot(want[0], want[1])
        if hs > self.max_speed:
            want[:2] *= self.max_speed / hs
        want[2] = np.clip(want[2], -self.climb, self.climb)
        dv = want - v
        dvn = np.linalg.norm(dv)
        if dvn > self.max_acc * dt:
            dv *= self.max_acc * dt / dvn
        v = v + dv
        p = p + v * dt
        if p[2] <= 0.0 and self._mode in ("landing",):
            p[2] = 0.0
            v = np.zeros(3)
            self._mode = "landed"
            self.tel.armed = self.tel.in_air = False
        p[2] = max(p[2], 0.0)
        if np.hypot(v[0], v[1]) > 0.5:
            self.tel.yaw = math.degrees(math.atan2(v[0], v[1])) % 360
        self.tel.pos, self.tel.vel = p, v

    async def takeoff(self, alt):
        self.tel.armed = self.tel.in_air = True
        self._mode = "offboard"
        self._sp_pos = np.array([self.tel.pos[0], self.tel.pos[1], alt])
        self._sp_vel = np.zeros(3)
        while self.tel.pos[2] < alt - 0.5:
            await self.clock.sleep(0.2)

    async def setpoint(self, pos, vel, yaw_deg=None):
        if self._mode in ("landed",):
            return
        self._mode = "offboard"
        self._sp_pos, self._sp_vel = np.asarray(pos, float), np.asarray(vel, float)

    async def land(self):
        if self._mode == "landed":
            return
        self._mode = "landing"
        while self._mode != "landed":
            await self.clock.sleep(0.2)

    async def return_home(self, alt):
        if self._mode == "landed":
            return
        self._rtl_alt = alt
        self._mode = "rtl"
        while self._mode != "landed":
            await self.clock.sleep(0.2)
