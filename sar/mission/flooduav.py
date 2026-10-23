"""FloodUAV backend: the two Unity drones as mission vehicles, and Unity's survivor finds as detections.

The Unity side runs in sim/flooduav/bridge.py: drone 0 is the build's search drone, drone 1
its pickup drone. Positions are mission meters with the origin at the search drone's start
point; each drone's home is its own pad.

Heights: the town has bridges and towers, so the bridge never flies below its min_alt
except to land. The vehicle adds alt_offset to every commanded height and subtracts it
from the reported one, so the orchestrator's 23 m search height becomes min_alt in Unity
and its settle and lag checks still close. Landing is real: the bridge descends straight
down and keeps the drone on the ground until the next takeoff.
"""

import asyncio
import math

import httpx
import numpy as np

from sar.mission.detect import Detection
from sar.mission.vehicle import SimClock, Vehicle


class FloodVehicle(Vehicle):
    def __init__(self, vid, clock: SimClock, bridge: str, alt_offset: float, camera_fov_deg: float = 60.0,
                 aspect: float = 16 / 9):
        super().__init__(vid, (0.0, 0.0), clock)
        self.bridge, self.alt_offset = bridge.rstrip("/"), alt_offset
        self.fov, self.aspect = math.radians(camera_fov_deg), aspect
        self.client = httpx.AsyncClient(timeout=3.0)
        self.landed = True
        self.state: dict = {}           # this drone's part of the bridge state
        self.found: list = []           # Unity's recorded survivors (shared by both drones)
        self.episode = None
        self.on_reset = None          # callback(reason) when Unity ends its mission and restarts
        self._task = None

    async def start(self):
        await self._poll()
        self.home = np.array([*self.state["pos"][:2], 0.0])
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
        await self.client.aclose()

    async def _poll(self):
        st = (await self.client.get(f"{self.bridge}/state")).json()
        if self.episode is not None and st["episode"] != self.episode:
            self.landed = True
            if self.on_reset:
                self.on_reset(st.get("last_end") or "Unity mission ended")
        self.episode, self.found = st["episode"], st["found"]
        self.state = d = st["drones"][self.id]
        p = np.array(d["pos"], float)
        p[2] = 0.0 if self.landed else max(p[2] - self.alt_offset, 0.0)
        self.tel.pos, self.tel.vel = p, np.array(d["vel"], float)
        self.tel.yaw = float(d["yaw"])
        self.tel.armed = self.tel.in_air = not self.landed
        self.tel.t_update = self.clock.now()

    async def _loop(self):
        while True:
            try:
                await self._poll()
            except (httpx.HTTPError, ValueError, KeyError, IndexError):
                pass          # no update: the orchestrator's link timeout notices
            await asyncio.sleep(0.1)

    async def _post(self, body):
        await self.client.post(f"{self.bridge}/setpoint", json={"drone": self.id, **body})

    async def _send(self, pos, vel=None):
        pos = np.asarray(pos, float).copy()
        pos[2] += self.alt_offset
        vel = np.asarray(vel if vel is not None else np.zeros(3), float)
        await self._post({"pos": pos.tolist(), "vel": vel.tolist()})

    async def takeoff(self, alt):
        self.landed = False
        await self._send([*self.tel.pos[:2], alt])
        while self.tel.pos[2] < alt - 1.0:
            await self.clock.sleep(0.2)

    async def setpoint(self, pos, vel, yaw_deg=None):
        if self.landed:
            return
        await self._send(pos, vel)

    async def land(self, timeout: float = 90.0):
        """Descend straight down; returns once the bridge reports the drone on the ground."""
        await self._post({"land": True})
        end = self.clock.now() + timeout
        while not self.state.get("landed") and self.clock.now() < end:
            await self.clock.sleep(0.3)
        self.landed = True

    async def return_home(self, alt):
        if self.landed:
            return
        await self._send([*self.home[:2], alt])
        while math.hypot(*(self.tel.pos[:2] - self.home[:2])) > 3.0 and not self.landed:
            await self.clock.sleep(0.2)
        await self.land()

    async def set_max_speed(self, v):
        await self.client.post(f"{self.bridge}/speed", json={"speed": float(v)})

    def footprint_m(self):
        """Half sizes of the downward camera's ground view; Unity's fov is the vertical one (along-track)."""
        h = self.state.get("agl") or (self.tel.pos[2] + self.alt_offset if self.tel.in_air else 0.0)
        along = max(float(h), 1.0) * math.tan(self.fov / 2)
        return [round(along, 1), round(along * self.aspect, 1)]


class FloodDetector:
    """Detections for both drones.

    The search drone (0) uses Unity's own finds: its visibility sensor confirms survivors up
    to 150 m away, and look() reports each found survivor while that drone is within `radius`,
    so a re-look from a displaced point sees it again and the orchestrator confirms it.
    The pickup drone has no sensor in the build, so it sees the survivors inside its camera
    footprint.
    """

    def __init__(self, vehicles: list[FloodVehicle], truth_en, radius: float = 150.0):
        self.vehicles, self.truth, self.radius = vehicles, np.asarray(truth_en, float), radius

    async def look(self, drone_id, pos, yaw_deg):
        v = self.vehicles[drone_id]
        if drone_id == 0:
            return [Detection(float(f["e"]), float(f["n"]), 0.95, drone_id) for f in v.found
                    if math.hypot(f["e"] - pos[0], f["n"] - pos[1]) <= self.radius]
        along, cross = v.footprint_m()
        r = min(along, cross)
        return [Detection(float(e), float(n), 0.9, drone_id) for e, n in self.truth
                if math.hypot(e - pos[0], n - pos[1]) <= r]


def fetch_info(bridge: str) -> dict:
    r = httpx.get(f"{bridge.rstrip('/')}/info", timeout=10.0)
    r.raise_for_status()
    return r.json()
