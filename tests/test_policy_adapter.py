"""Search policy adapter and detector service helpers."""

import numpy as np
import pytest

from sar.mission.search_policy import AgentView, SearchPolicy, grid_for_extent


def world(g=32):
    mask = np.zeros((g, g), bool)
    mask[4:28, 4:28] = True
    belief = np.where(mask, 1.0 / mask.sum(), 0.0) * 3
    return mask, belief


@pytest.mark.parametrize("mode", ["greedy", "lawnmower"])
def test_decide_returns_contiguous_segment(mode):
    pol = SearchPolicy(32, None)
    mask, belief = world()
    agents = [AgentView(True, (10, 10)), AgentView(True, (20, 20)), AgentView(False, (0, 0))]
    a, cells = pol.decide(mode, 0, belief, mask, np.zeros_like(mask, np.int32), agents, (16, 16), 0.8, 3.0, 0.1)
    assert len(cells) in (1, 4)
    prev = np.array([10, 10])
    for c in cells if len(cells) == 4 else []:
        assert np.abs(np.array(c) - prev).max() == 1      # 8-connected moves
        prev = np.array(c)


def test_separation_mask_blocks_close_segments():
    pol = SearchPolicy(32, None)
    agents = [AgentView(True, (10, 10)), AgentView(True, (10, 16), seg=[(10, 15), (10, 14)]), AgentView(False, (0, 0))]
    extra = np.asarray(pol.separation_mask(agents, 0))
    ends = np.array([10, 10]) + np.asarray(pol.env.offsets)[:, -1]
    east = int(np.argmin(np.abs(ends - np.array([10, 14])).sum(-1)))
    assert not extra[0, east]
    assert extra[0, pol.env.hold]


def test_escape_when_drones_touch():
    pol = SearchPolicy(32, None)
    mask, belief = world()
    agents = [AgentView(True, (15, 15)), AgentView(True, (15, 16), seg=[(15, 16)]), AgentView(False, (0, 0))]
    a, cells = pol.decide("greedy", 0, belief, mask, np.zeros_like(mask, np.int32), agents, (16, 16), 0.8, 3.0, 0.1)
    assert len(cells) == 4
    assert np.abs(np.array(cells[-1]) - np.array([15, 16])).max() > 2


def test_grid_for_extent():
    assert grid_for_extent(20) == 32 and grid_for_extent(40) == 64 and grid_for_extent(90) == 96
    with pytest.raises(ValueError):
        grid_for_extent(200)


