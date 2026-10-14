"""Orchestrator tests with kinematic vehicles at high sim speed (plan section 13)."""

import asyncio
import math

import numpy as np
import pytest

from sar.config import MissionConfig
from sar.mission.detect import SimDetector
from sar.mission.geo import AreaGrid, Frame, koester_prior
from sar.mission.orchestrator import Orchestrator
from sar.mission.vehicle import KinematicVehicle, SimClock

SPEED = 100.0
HOME = (149.58, -33.42)


def square(d=0.004, c=HOME):
    return [[c[0] - d, c[1] - d], [c[0] + d, c[1] - d], [c[0] + d, c[1] + d], [c[0] - d, c[1] + d]]


def make(roles=("search", "search", "rescue"), truth=None, fp=0.0, pd=0.9, footprint_m=None, **cfg_kw):
    cfg = MissionConfig(pd=pd, **cfg_kw)
    clock = SimClock(SPEED)
    pads = [(6.0 * (i - 1), 0.0) for i in range(len(roles))]
    vs = [KinematicVehicle(i, pads[i], clock, max_speed=cfg.transit_speed) for i in range(len(roles))]

    def det(orch, truth_en):
        t = np.zeros((0, 2)) if truth_en is None else truth_en
        return SimDetector(t, pd=pd, fp=fp, footprint_m=footprint_m or cfg.cell_m, seed=1)

    return Orchestrator(cfg, vs, list(roles), clock, None, det, seed=0)


async def run_until(orch, cond, t_max):
    t0 = orch.now()
    while not cond():
        if orch.now() - t0 > t_max:
            return False
        await asyncio.sleep(0.02)
    return True


def test_geo_roundtrip():
    f = Frame(*HOME)
    g = AreaGrid(f, square(), 30.0, 32)
    e, n = g.cell_center(5, 7)
    assert g.cell_of(e, n) == (5, 7)
    lon, lat = f.to_lonlat(*f.to_en(150.0, -33.0))
    assert abs(lon - 150.0) < 1e-9 and abs(lat + 33.0) < 1e-9
    # 0.008 deg square is about 744 m x 890 m
    area = g.mask.sum() * 900.0
    assert 0.55e6 < area < 0.8e6
    p = koester_prior(g, (0.0, 0.0))
    assert abs(p.sum() - 1) < 1e-6 and p[~g.mask].sum() == 0


def test_search_separation_and_finds():
    async def main():
        orch = make()
        await orch.start()
        f = orch.frame
        truth = np.array([[100.0, 120.0], [-150.0, -60.0]])
        orch.start_mission(polygon=square(), mode="planner", n_search=2, expected_survivors=2, truth_en=truth)
        min_d = 1e9
        t0 = orch.now()
        while orch.now() - t0 < 900:
            await asyncio.sleep(0.01)
            a, b = orch.drones[0], orch.drones[1]
            if a.state == b.state == "SEARCH" and a.plan_cell and b.plan_cell:
                min_d = min(min_d, float(np.linalg.norm(a.vehicle.tel.pos[:2] - b.vehicle.tel.pos[:2])))
        m = orch.mission
        assert orch.metrics["looks"] > 100
        assert m.belief.sum() < 0.8 * m.belief0
        assert min_d > orch.cfg.cell_m, min_d
        st = [s.status for s in orch.survivors.values()]
        assert any(s in ("confirmed", "assigned", "reached") for s in st)
        await orch.shutdown()
    asyncio.run(main())


def test_recall_and_replace():
    async def main():
        orch = make(roles=("search", "search", "search"))
        await orch.start()
        orch.start_mission(polygon=square(), mode="lawnmower", n_search=2, expected_survivors=1,
                           options={"auto_replace": True})
        assert await run_until(orch, lambda: orch.drones[0].plan_cell is not None, 300)
        await orch.command({"type": "recall", "drone": 0})
        assert orch.drones[0].state == "RETURNING"
        assert orch.drones[2].state == "SEARCH"
        assert await run_until(orch, lambda: orch.drones[0].state == "LANDED", 600)
        assert not orch.drones[0].vehicle.tel.in_air
        await orch.shutdown()
    asyncio.run(main())


def test_lost_link_aborts_then_lands():
    async def main():
        orch = make()
        await orch.start()
        orch.start_mission(polygon=square(), mode="greedy", n_search=1, expected_survivors=1)
        assert await run_until(orch, lambda: orch.drones[0].plan_cell is not None, 300)
        orch.drones[0].vehicle.link_ok = False
        assert await run_until(orch, lambda: orch.drones[0].state == "ABORT", 20)
        orch.drones[0].vehicle.link_ok = True
        assert await run_until(orch, lambda: orch.drones[0].state == "LANDED", 600)
        await orch.shutdown()
    asyncio.run(main())


def test_battery_rule():
    async def main():
        orch = make(battery_s=400.0)
        await orch.start()
        orch.start_mission(polygon=square(), mode="planner", n_search=1, expected_survivors=1)
        assert await run_until(orch, lambda: orch.drones[0].state == "RETURNING", 500)
        d = orch.drones[0]
        dist = float(np.hypot(*(d.vehicle.tel.pos[:2] - d.vehicle.home[:2])))
        assert d.battery_s >= dist / orch.cfg.transit_speed + orch.cfg.reserve_s - 5
        assert await run_until(orch, lambda: d.state == "LANDED", 600)
        assert d.battery_s > 0
        await orch.shutdown()
    asyncio.run(main())


def test_operator_dispatch_and_standby_no_drain():
    async def main():
        # A wide footprint: looks are taken along the path, and with one cell the route
        # (which depends on timing) can pass the survivor between two looks.
        orch = make(fp=0.0, footprint_m=60.0)
        await orch.start()
        truth = np.array([[60.0, 60.0]])
        orch.start_mission(polygon=square(), mode="planner", n_search=1, expected_survivors=1, truth_en=truth,
                           options={"auto_dispatch": False})
        r = orch.drones[2]
        b0 = r.battery_s
        assert await run_until(orch, lambda: any(s.status != "candidate" or len(s.views) for s in orch.survivors.values()),
                               1500), [e["text"] for e in orch.events[-12:]]
        s = next(iter(orch.survivors.values()))
        assert r.state == "STANDBY" and abs(r.battery_s - b0) < 1e-6   # landed standby uses no flight time
        await orch.command({"type": "dispatch", "drone": 2, "survivor": s.id})
        assert r.state == "RESCUE_TRANSIT"
        assert await run_until(orch, lambda: s.status == "reached", 600)
        assert math.hypot(r.vehicle.tel.pos[0] - s.e, r.vehicle.tel.pos[1] - s.n) < 5
        assert await run_until(orch, lambda: r.state == "STANDBY" and not r.vehicle.tel.in_air, 600)
        await orch.shutdown()
    asyncio.run(main())


def test_bad_commands_raise():
    async def main():
        orch = make()
        await orch.start()
        with pytest.raises(ValueError):
            orch.start_mission(polygon=square(0.05), mode="planner")      # too large
        with pytest.raises(ValueError):
            orch.start_mission(polygon=square(), mode="nope")
        with pytest.raises(ValueError):
            await orch.command({"type": "bogus"})
        await orch.shutdown()
    asyncio.run(main())
