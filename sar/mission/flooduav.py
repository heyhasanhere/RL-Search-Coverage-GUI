"""FloodUAV backend: the two Unity drones as mission vehicles, and box detections of the true survivors.

The Unity side runs in sim/flooduav/bridge.py: drone 0 is the build's search drone, drone 1
its pickup drone. Positions are mission meters with the origin at the search drone's start
point; each drone's home is its own pad.

Heights: the town has bridges and towers, so the bridge never flies below its min_alt
except to land. The vehicle adds alt_offset to every commanded height and subtracts it
from the reported one, so the orchestrator's 23 m search height becomes min_alt in Unity
and its settle and lag checks still close. Landing is real: the bridge descends straight
down and keeps the drone on the ground until the next takeoff.

The search drone can also be flown by the build's own trained agent (fly_agent, the
orchestrator's rl_unity mode); the next setpoint, hold or landing takes it back.
"""

import asyncio
import math

import httpx
import numpy as np

from sar.mission.detect import Detection
from sar.mission.vehicle import SimClock, Vehicle


class FloodVehicle(Vehicle):
    def __init__(self, vid, clock: SimClock, bridge: str, alt_offset: float, camera_fov_deg: float = 60.0,
                 aspect: float = 16 / 9, view_alt: float = 40.0, track_lead: float = 0.0):
        super().__init__(vid, (0.0, 0.0), clock)
        self.track_lead = track_lead    # the bridge steers from the drone's position this far ahead
        self.bridge, self.alt_offset = bridge.rstrip("/"), alt_offset
        # the detection box: the downward camera's ground view from view_alt (Unity's fov is the
        # vertical one, along-track), fixed whatever the drone's actual height
        along = view_alt * math.tan(math.radians(camera_fov_deg) / 2)
        self.box = [round(along, 1), round(along * aspect, 1)]
        self.client = httpx.AsyncClient(timeout=3.0)
        self.landed = True
        self.state: dict = {}           # this drone's part of the bridge state
        self.found: list = []           # Unity's recorded survivors (shared by both drones)
        self.episode = None
        self.on_reset = None          # callback(reason) when Unity ends its mission and restarts
        self.agent_capable = False    # the bridge has the agent and this is its drone (build_flooduav)
        self.agent_flying = False     # the bridge reports the agent flying this drone
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
        self.agent_flying = self.id == 0 and st.get("pilot") == "onnx"
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

    async def _post(self, body, tries=3):
        """One command to the bridge; a slow or lost reply is sent again before it fails the task."""
        for k in range(tries):
            try:
                await self.client.post(f"{self.bridge}/setpoint", json={"drone": self.id, **body})
                return
            except httpx.HTTPError:
                if k == tries - 1:
                    raise

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

    async def fly_agent(self, limits="training", agent=None):
        """A trained agent (the bridge's default: None) flies this drone, from wherever it is (it takes off by itself)."""
        msg = {"pilot": "onnx", "limits": limits} | ({"agent": agent} if agent else {})
        r = await self.client.post(f"{self.bridge}/pilot", json=msg)
        if r.status_code != 200:
            raise RuntimeError(r.json().get("error", r.text))
        self.landed, self.agent_flying = False, True

    @property
    def agl(self):
        """Height above whatever is below, m; None: nothing within the ray's range."""
        return self.state.get("agl")

    async def set_max_speed(self, v):
        await self.client.post(f"{self.bridge}/speed", json={"speed": float(v)})

    def footprint_m(self):
        """Half sizes of the detection box, m: [along-track, cross-track]."""
        return self.box

    def in_view(self, pos, yaw_deg, e, n):
        """(e, n) inside footprint_m() around `pos`, along-track pointing to compass heading `yaw_deg`."""
        along, cross = self.footprint_m()
        h = math.radians(yaw_deg)
        de, dn = e - pos[0], n - pos[1]
        a = de * math.sin(h) + dn * math.cos(h)
        c = de * math.cos(h) - dn * math.sin(h)
        return abs(a) <= along and abs(c) <= cross


class FloodDetector:
    """Detections for both drones: every survivor inside the drone's detection box (footprint_m()).

    The rule uses the true survivor positions and no line of sight, so the box on the map is
    exactly where a drone detects. Unity's own sensor is not used.
    """

    def __init__(self, vehicles: list[FloodVehicle], truth_en):
        self.vehicles, self.truth = vehicles, np.asarray(truth_en, float)

    async def look(self, drone_id, pos, yaw_deg):
        v = self.vehicles[drone_id]
        return [Detection(float(e), float(n), 0.9, drone_id) for e, n in self.truth
                if v.in_view(pos, yaw_deg, e, n)]


def fetch_info(bridge: str) -> dict:
    r = httpx.get(f"{bridge.rstrip('/')}/info", timeout=10.0)
    r.raise_for_status()
    return r.json()
