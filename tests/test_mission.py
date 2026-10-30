"""Orchestrator tests with kinematic vehicles at high sim speed (plan section 13)."""

import asyncio
import math

import numpy as np
import pytest

from sar.config import MissionConfig
from sar.mission.detect import Detection, SimDetector
from sar.mission.geo import AreaGrid, Frame, koester_prior
from sar.mission.orchestrator import Orchestrator, Survivor
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


def test_reset_numbers_survivors_from_one():
    async def main():
        orch = make(fp=1.0)                    # every look reports a candidate: survivors appear at once
        await orch.start()
        start = dict(polygon=square(), mode="planner", n_search=2, expected_survivors=2)
        orch.start_mission(**start)
        assert await run_until(orch, lambda: orch.survivors, 300)
        assert min(orch.survivors) == 1
        await orch.command({"type": "reset_mission"})
        assert orch.mission is None and not orch.survivors
        assert all(d.state == "IDLE" and not d.vehicle.tel.in_air for d in orch.drones)
        orch.start_mission(**start)
        assert await run_until(orch, lambda: orch.survivors, 300)
        assert min(orch.survivors) == 1
        await orch.shutdown()
    asyncio.run(main())

def test_new_survivors_resets_then_lays_out():
    async def main():
        orch = make(fp=1.0)
        await orch.start()
        calls = []

        async def layout(count, mix):
            calls.append((orch.mission, count, mix))    # reset_mission ran first
            return 42, count
        with pytest.raises(ValueError):           # the kinematic backend has no layouts
            await orch.command({"type": "new_survivors"})
        orch.on_new_survivors = layout
        orch.start_mission(polygon=square(), mode="planner", n_search=2, expected_survivors=2)
        assert await run_until(orch, lambda: orch.survivors, 300)
        await orch.command({"type": "new_survivors", "count": 30})
        assert calls == [(None, 30, None)] and orch.mission is None and not orch.survivors
        with pytest.raises(ValueError):
            await orch.command({"type": "new_survivors", "count": 0})
        await orch.command({"type": "new_survivors", "mix": {"roofs": 80, "water": 20}})
        assert calls[-1][2] == {"roofs": 80.0, "high_ground": 0.0, "bridges": 0.0, "water": 20.0, "uniform": 0.0}
        for bad in ({"roofs": 0}, {"roofs": -1, "water": 5}, {"lake": 1}, [50, 50]):
            with pytest.raises(ValueError):
                await orch.command({"type": "new_survivors", "mix": bad})
        assert all(d.state == "IDLE" and not d.vehicle.tel.in_air for d in orch.drones)
        await orch.shutdown()
    asyncio.run(main())

def test_continue_keeps_search_state():
    async def main():
        orch = make(fp=1.0)
        await orch.start()
        with pytest.raises(ValueError):
            await orch.command({"type": "continue_mission"})
        orch.start_mission(polygon=square(), mode="planner", n_search=2, expected_survivors=2)
        assert await run_until(orch, lambda: orch.survivors and orch.metrics["looks"] > 20, 300)
        with pytest.raises(ValueError):           # still running
            await orch.command({"type": "continue_mission"})
        await orch.command({"type": "stop_mission"})
        assert await run_until(orch, lambda: orch.mission.status == "finished", 600)
        m, looks, found = orch.mission, orch.mission.looks.copy(), set(orch.survivors)
        await orch.command({"type": "continue_mission"})
        assert orch.mission is m and m.status == "running" and not m.stopping
        assert found <= set(orch.survivors) and (m.looks >= looks).all()
        assert [d.state for d in orch.drones[:2]] == ["SEARCH", "SEARCH"]
        assert await run_until(orch, lambda: m.looks.sum() > looks.sum(), 300)
        await orch.shutdown()
    asyncio.run(main())

def test_rescue_flight_looks_and_detects():
    async def main():
        orch = make(roles=("search", "rescue"), pd=1.0)
        await orch.start()
        truth = np.array([[150.0, 150.0]])              # on the rescue drone's way to the dispatched survivor
        orch.start_mission(polygon=square(), mode="lawnmower", n_search=1, expected_survivors=1, truth_en=truth)
        seen = []
        on_detection = orch._on_detection

        def record(d, det, viewpoint):
            seen.append((d.id, d.state, det.e, det.n))
            on_detection(d, det, viewpoint)
        orch._on_detection = record
        far = Survivor(orch._next_sid, 300.0, 300.0, 0.9, orch.t_mission(), status="confirmed")
        orch._next_sid += 1
        orch.survivors[far.id] = far
        await orch.command({"type": "dispatch", "drone": 1, "survivor": far.id})
        assert await run_until(orch, lambda: far.status == "reached", 600)
        assert any(i == 1 and st == "RESCUE_TRANSIT" and math.hypot(e - 150, n - 150) < 10 for i, st, e, n in seen)
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


def test_auto_switch_sends_near_search_drone():
    async def main():
        orch = make()
        await orch.start()
        orch.start_mission(polygon=square(), mode="lawnmower", n_search=2, expected_survivors=1)
        d = orch.drones[0]
        assert await run_until(orch, lambda: d.plan_cell is not None, 300)
        p = d.vehicle.tel.pos
        # the rescue drone is busy far away: waiting for it is slower than switching d
        r = orch.drones[2]
        r.state, r.target = "RESCUE_TRANSIT", np.array([-2000.0, -2000.0])
        s = Survivor(orch._next_sid, float(p[0]) + 20.0, float(p[1]), 0.9, orch.t_mission(), status="confirmed")
        orch._next_sid += 1
        orch.survivors[s.id] = s
        assert await run_until(orch, lambda: s.status != "confirmed", 10)
        assert s.assigned == d.id and d.resume_search
        r.state, r.target = "STANDBY", None
        assert await run_until(orch, lambda: s.status == "reached", 300)
        assert await run_until(orch, lambda: d.state == "SEARCH", 30)
        await orch.shutdown()
    asyncio.run(main())


def test_auto_switch_waits_for_near_rescue_drone():
    async def main():
        orch = make()
        await orch.start()
        orch.start_mission(polygon=square(), mode="lawnmower", n_search=2, expected_survivors=1,
                           options={"auto_dispatch": False})
        assert await run_until(orch, lambda: orch.drones[0].plan_cell is not None, 300)
        # next to the rescue drone's pad: no search drone switches
        s = Survivor(orch._next_sid, 6.0, 5.0, 0.9, orch.t_mission(), status="confirmed")
        orch._next_sid += 1
        orch.survivors[s.id] = s
        await run_until(orch, lambda: False, 5)
        assert s.status == "confirmed"
        orch.drones[2].state = "RETURNING"         # no rescue drone left: a search drone would switch now
        orch.mission.mode = "rl_unity"             # but the Unity agent's mode has its own rules
        orch._auto_switch()
        assert s.status == "confirmed"
        orch.mission.mode = "lawnmower"
        orch._auto_switch()
        assert s.status == "assigned"
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


def test_look_covers_footprint_once_per_pass():
    async def main():
        orch = make(roles=("search", "rescue"))
        await orch.start()
        orch.start_mission(polygon=square(), mode="lawnmower", n_search=0, expected_survivors=1)
        m, d = orch.mission, orch.drones[0]
        d.vehicle.footprint_m = lambda: [45.0, 45.0]     # a 90 m box: 3 x 3 cells of 30 m
        cell = m.grid.cell_of(0.0, 0.0)
        e, n = m.grid.cell_center(*cell)
        d.vehicle.tel.pos[:2] = [e, n]
        b0 = m.belief.copy()
        orch._look(d, cell)
        assert int(((m.looks > 0) & m.grid.mask).sum()) == 9
        assert np.allclose(m.belief[m.looks > 0], b0[m.looks > 0] * (1 - orch.cfg.pd))
        orch._look(d, cell)                               # still in view: no second update
        assert m.looks.max() == 1
        orch._look(d, cell, again=True)                   # a deliberate hold looks again
        assert m.looks.max() == 2
        await orch.shutdown()
    asyncio.run(main())


def test_relook_option():
    async def main():
        orch = make(roles=("search", "rescue"))
        await orch.start()
        orch.start_mission(polygon=square(), mode="lawnmower", n_search=1, expected_survivors=1,
                           options={"auto_dispatch": False, "relook": False})
        d = orch.drones[0]
        d.state = "SEARCH"
        orch._on_detection(d, Detection(100.0, 100.0, 0.9, d.id), np.array([90.0, 90.0, 30.0]))
        s = next(iter(orch.survivors.values()))
        assert s.status == "confirmed" and d.relook is None          # off: confirmed on the first detection
        await orch.command({"type": "set_options", "relook": True})
        assert orch.state_msg()["options"]["relook"]
        orch._on_detection(d, Detection(-100.0, -100.0, 0.9, d.id), np.array([-90.0, -90.0, 30.0]))
        s2 = orch.survivors[d.relook["sid"]]
        assert s2.status == "candidate"                               # on: held for the re-look
        await orch.shutdown()
    asyncio.run(main())


def test_close_survivors_merge():
    async def main():
        for merge_m, found in ((8.0, 1), (0.5, 2)):   # default merge, FloodUAV's exact-position merge
            orch = make(roles=("search", "rescue"), merge_m=merge_m)
            await orch.start()
            orch.start_mission(polygon=square(), mode="lawnmower", n_search=1, expected_survivors=2,
                               options={"auto_dispatch": False, "relook": False})
            d = orch.drones[0]
            d.state = "SEARCH"
            vp = np.array([90.0, 90.0, 30.0])
            for e, n in ((100.0, 100.0), (101.0, 101.5), (100.0, 100.0)):   # two survivors 1.8 m apart, one seen twice
                orch._on_detection(d, Detection(e, n, 0.9, d.id), vp)
            assert len(orch.survivors) == found
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


class AgentVehicle(KinematicVehicle):
    """A kinematic drone that the sim's trained agent can fly (rl_unity); it records the agent calls."""
    agent_capable, agl = True, 30.0

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.agent_calls, self.agent_names, self.agent_flying = [], [], False

    async def fly_agent(self, limits="training", agent=None):
        self.agent_calls.append(limits)
        self.agent_names.append(agent)
        self.agent_flying = True


def test_agent_mode_flies_and_hands_back():
    async def main():
        cfg = MissionConfig(pd=0.9)
        clock = SimClock(SPEED)
        vs = [AgentVehicle(0, (0.0, 0.0), clock, max_speed=cfg.transit_speed),
              KinematicVehicle(1, (6.0, 0.0), clock, max_speed=cfg.transit_speed)]
        orch = Orchestrator(cfg, vs, ["search", "rescue"], clock, None,
                            lambda o, t: SimDetector(np.zeros((0, 2)), pd=0.9, fp=0.0, footprint_m=cfg.cell_m, seed=1))
        assert "rl_unity" in orch.modes
        await orch.start()
        try:
            orch.start_mission(polygon=square(), lkp=list(HOME), mode="rl_unity", n_search=1)
            d = orch.drones[0]
            assert await run_until(orch, lambda: d.agent, 30)
            assert d.task_text == "searching (Unity agent)" and vs[0].agent_calls == ["bridge"]
            await orch.command({"type": "set_agent_limits", "limits": "training"})
            assert vs[0].agent_calls[-1] == "training" and orch.state_msg()["agent_limits"] == "training"
            orch.agents, orch.agent = [{"id": "ppo", "label": "PPO 19M"}, {"id": "dqn", "label": "DQN 5M"}], "ppo"
            await orch.command({"type": "set_agent", "agent": "dqn"})
            assert vs[0].agent_names[-1] == "dqn" and orch.state_msg()["agent"] == "dqn"
            with pytest.raises(ValueError):
                await orch.command({"type": "set_agent", "agent": "nope"})
            await orch.command({"type": "recall", "drone": 0})
            assert await run_until(orch, lambda: not d.agent, 5)
            assert d.state in ("RETURNING", "LANDED")      # the fake agent never took off
        finally:
            await orch.shutdown()
        # backends without the agent do not offer the mode
        assert "rl_unity" not in make().modes
    asyncio.run(main())
