"""M4 search modes in the orchestrator: `mcts` (hand search) and `rl_search` (network).

Short kinematic missions (plan section 13 pattern): drones leave their entry
cell, every planned segment stays inside the area, and the two search drones
keep the separation rule.
"""

import asyncio

import numpy as np
import pytest

from sar.mission.search_policy import MODES, AgentView
from sar.mission.search_policy_v2 import MODES_V2, get_policy_v2
from test_mission import make, square

NEED_NET = pytest.mark.skipif(not get_policy_v2(32).has_net, reason="M4 checkpoint missing")


def test_modes_listed():
    assert set(MODES_V2) <= set(MODES)


@pytest.mark.parametrize("mode", ["mcts", pytest.param("rl_search", marks=NEED_NET)])
def test_mission_v2_mode(mode):
    async def main():
        orch = make()
        await orch.start()
        orch.start_mission(polygon=square(), mode=mode, n_search=2, expected_survivors=2)
        m = orch.mission
        g = m.grid
        # dilated area: the env allows segment ends one cell outside the mask (polygon edge)
        dil = g.mask.copy()
        dil[1:] |= g.mask[:-1]; dil[:-1] |= g.mask[1:]
        dil[:, 1:] |= dil[:, :-1].copy(); dil[:, :-1] |= dil[:, 1:].copy()
        segs, min_d, decisions = [], 1e9, []
        orig = orch._decide

        def spy(d, from_cell, warm=False):
            out = orig(d, from_cell, warm)
            if not warm:
                decisions.append((d.id, out))
            return out

        orch._decide = spy
        t0 = orch.now()
        start = {d.id: None for d in orch.drones[:2]}
        while orch.now() - t0 < 400:
            await asyncio.sleep(0.01)
            a, b = orch.drones[0], orch.drones[1]
            for d in (a, b):
                if d.state == "SEARCH" and d.plan_cell and start[d.id] is None:
                    start[d.id] = d.vehicle.tel.pos[:2].copy()
            if a.state == b.state == "SEARCH" and a.plan_cell and b.plan_cell:
                min_d = min(min_d, float(np.linalg.norm(a.vehicle.tel.pos[:2] - b.vehicle.tel.pos[:2])))
        moved = [d for d in orch.drones[:2]
                 if start[d.id] is not None and np.linalg.norm(d.vehicle.tel.pos[:2] - start[d.id]) > 60]
        assert len(moved) == 2, "both search drones should fly segments"
        moves = [o for _, o in decisions if len(o[1]) > 1]
        assert len(moves) > 10
        for a, cells, alt in moves:
            assert alt == orch.cfg.search_alt
            for r, c in cells:
                assert 0 <= r < g.g and 0 <= c < g.g and dil[r, c], (r, c)
        assert orch.metrics["looks"] > 40
        assert m.belief.sum() < m.belief0
        assert min_d > orch.cfg.cell_m, min_d
        await orch.shutdown()
    asyncio.run(main())


@pytest.mark.parametrize("mode", ["mcts", pytest.param("rl_search", marks=NEED_NET)])
def test_v2_separation_and_timing(mode):
    """Another drone's segment blocks nearby actions; decisions stay under 200 ms."""
    import time

    g = 32
    pol = get_policy_v2(g)
    mask = np.zeros((g, g), bool)
    mask[2:g - 2, 2:g - 2] = True
    belief = np.where(mask, 1.0, 0.0) * 3 / mask.sum()
    belief[16, 20:24] = 1.0                  # the prize sits on the other drone's path
    other_seg = [(16, 21), (16, 22), (16, 23), (16, 24)]
    agents = [AgentView(True, (16, 16)), AgentView(True, (16, 20), seg=other_seg), AgentView(False, (0, 0))]
    looks = np.zeros((g, g), np.int32)
    pol.decide(mode, 0, belief, mask, looks, agents, (16, 16), 0.8, float(belief.sum()), 0.1)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        a, cells, alt = pol.decide(mode, 0, belief, mask, looks, agents, (16, 16), 0.8, float(belief.sum()), 0.1)
        ts.append(time.perf_counter() - t)
        others = np.asarray(other_seg + [(16, 20)])
        for c in cells:
            assert np.abs(others - np.asarray(c)).max(-1).min() > pol.cfg.min_sep
    assert np.median(ts) < 0.2, ts
