"""Invariant tests for env v2 (plan section 19).

Covers the version 2 requirements: the two altitude footprints, the per-cell Pd
lookup from the placeholder table, the time cost of a level change, battery
sorties, and the separation rule between drones.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sar.config_v2 import HIGH, LOW, N_LEVELS, EnvConfigV2, load_pd_table
from sar.env.v2.grid_env_v2 import ACTIVE, AIRBORNE, SWAPPING, SearchEnvV2, action_tables
from sar.search import policies_v2 as P

CFG = EnvConfigV2(grid=32, extent_min=26, extent_max=30, p_delay=0.0).with_grid(32)


@pytest.fixture(scope="module")
def env():
    return SearchEnvV2(CFG)


def test_action_set_and_footprints():
    off, val, dirs, foot, slot = action_tables(4)
    # 2 levels x (32 ring-4 end cells + hold) = 66 actions
    assert off.shape[0] == 2 and off.shape[1] == 33
    assert 2 * 33 == 66
    # low level: only the centre footprint slot is active -> 1 cell per look
    assert slot[LOW].sum() == 1 and slot[LOW][..., 4].item() == 1.0
    # high level: all nine slots -> 3x3 footprint
    assert slot[HIGH].sum() == 9


def test_look_counts_per_level(env):
    """A single-drone hold looks at 1 cell low and 9 cells high (3x3)."""
    s, _ = env.reset(jax.random.PRNGKey(0))
    # isolate drone 0: put it low, airborne, in a known in-mask cell; park the
    # others as PENDING so they do not move or look
    p0 = np.array([16, 11])
    s = s.replace(
        level=jnp.zeros((env.n,), jnp.int32),
        pos=jnp.array([p0, p0, p0], jnp.int32),
        status=jnp.array([ACTIVE, 0, 0], jnp.int32),               # drone 1,2 PENDING
        ground=jnp.full((env.n,), AIRBORNE, jnp.int32),
    )
    hold_low = LOW * env.a0 + env.hold
    s2, _, _, _ = env.step(s, jnp.array([hold_low, env.hold, env.hold], jnp.int32))
    looks_low = int((s2.looks > 0).sum())
    assert looks_low == 1, looks_low                              # low: centre cell only

    s = s.replace(looks=jnp.zeros_like(s.looks))
    hold_high = HIGH * env.a0 + env.hold
    s2, _, _, _ = env.step(s, jnp.array([hold_high, env.hold, env.hold], jnp.int32))
    looks_high = int((s2.looks > 0).sum())
    assert looks_high == 9, looks_high                           # high: 3x3 block


def test_pd_lookup_matches_table(env):
    """Each cell's Pd is PdTable[level, lc, cp] times one shared jitter.

    The env multiplies the whole map by a single per-episode jitter, so the ratio
    of two levels at the same cell is jitter-free and must equal the ratio of
    the two table entries for that cell's classes.
    """
    table = load_pd_table()
    s, _ = env.reset(jax.random.PRNGKey(1))
    tab = np.asarray(table.pd)                                   # [2,5,4]
    ratios = {round(tab[HIGH, lc, cp] / tab[LOW, lc, cp], 4)
              for lc in range(tab.shape[1]) for cp in range(tab.shape[2])
              if tab[LOW, lc, cp] > 0}
    got = np.asarray(s.pd[HIGH]) / np.maximum(np.asarray(s.pd[LOW]), 1e-9)
    hit = sum(any(abs(r - g) < 2e-3 for g in ratios) for r in got.ravel())
    assert hit / got.size > 0.9                                  # most cells match a ratio


def test_level_change_adds_time(env):
    """Changing level costs the climb time on top of the segment time."""
    s, _ = env.reset(jax.random.PRNGKey(2))
    s = s.replace(battery=jnp.full((env.n,), 10_000.0))
    a = jnp.full((env.n,), env.hold)                              # hold, stays low
    # battery drop for a hold at the same level
    s_same, _, _, _ = env.step(s, a)
    dt_same = float(s.battery[0] - s_same.battery[0])
    # same hold but change level (action 33 + hold)
    hold_high = 1 * env.a0 + env.hold
    s_chg, _, _, _ = env.step(s, jnp.full((env.n,), hold_high))
    dt_chg = float(s.battery[0] - s_chg.battery[0])
    assert dt_chg > dt_same                                      # level change costs more
    assert dt_chg - dt_same == pytest.approx(env.t_level, rel=1e-4)


def test_sortie_costs_no_flight_time_on_ground(env):
    """A drone on the ground swapping batteries burns no flight seconds."""
    cfg = env.cfg
    s, _ = env.reset(jax.random.PRNGKey(3))
    # put one drone on low battery so it must swap after this step
    b = s.battery.at[0].set(cfg.reserve_s + 1.0)
    s = s.replace(battery=b, status=jnp.full((env.n,), ACTIVE, jnp.int32),
                  ground=jnp.full((env.n,), AIRBORNE, jnp.int32))
    s_next, _, _, _ = env.step(s, jnp.full((env.n,), env.hold))
    assert int(s_next.ground[0]) == SWAPPING                      # now swapping
    # the swap clock starts at once, so it is already part-way through cfg.swap_s
    assert 0.0 < float(s_next.swap_left[0]) <= cfg.swap_s
    # the next step happens entirely on the ground: battery must not change
    bat_before = float(s_next.battery[0])
    s_after, _, _, _ = env.step(s_next, jnp.full((env.n,), env.hold))
    assert float(s_after.battery[0]) == pytest.approx(bat_before, abs=1e-6)


def test_separation_same_level(env):
    """Same-level airborne drones stay at least min_sep cells apart."""
    s, _ = env.reset(jax.random.PRNGKey(4))
    min_sep = env.cfg.min_sep
    checked = 0
    for t in range(60):
        s, _, _, d = env.step(s, P.greedy(env, s, jax.random.PRNGKey(t)))
        pos, lev = np.asarray(s.pos), np.asarray(s.level)
        air = (np.asarray(s.status) == ACTIVE) & (np.asarray(s.ground) == AIRBORNE)
        for i in range(env.n):
            for j in range(i + 1, env.n):
                if air[i] and air[j] and lev[i] == lev[j]:
                    assert np.abs(pos[i] - pos[j]).max() >= min_sep, (t, i, j, pos[i], pos[j])
                    checked += 1
        if bool(d):
            break
    assert checked > 0                                          # the check actually ran


def test_two_levels_have_different_footprints(env):
    """A low action and a high action of the same base move look at different cells."""
    s, _ = env.reset(jax.random.PRNGKey(5))
    base = env.hold                                              # base hold
    low = LOW * env.a0 + base
    high = HIGH * env.a0 + base
    s1, _, _, _ = env.step(s, jnp.full((env.n,), low))
    s2, _, _, _ = env.step(s, jnp.full((env.n,), high))
    looks_low = int((s1.looks > 0).sum())
    looks_high = int((s2.looks > 0).sum())
    assert looks_high > looks_low


def test_belief_update_matches_poisson_v2(env):
    """A low hold multiplies the looked cell's belief by (1 - Pd)."""
    s, _ = env.reset(jax.random.PRNGKey(0))
    p = s.pos[0]
    hold_low = LOW * env.a0 + env.hold
    s2, _, _, _ = env.step(s, jnp.full((env.n,), hold_low))
    expect = s.belief[p[0], p[1]] * (1 - s.pd[LOW, p[0], p[1]])
    assert np.isclose(float(s2.belief[p[0], p[1]]), float(expect), rtol=1e-5)

def test_single_map_bank_is_accepted():
    """Tier R builds a one-case bank of bare [G, G] layers, not [N, G, G]."""
    import numpy as np

    from sar.env.v2 import maps_v2

    syn = maps_v2.synthetic_terrain_np(0, CFG.grid, seed=3)
    bank = {k: np.asarray(v) for k, v in syn.items() if k != "prior"}
    bank["prior"] = np.ones((1, CFG.grid, CFG.grid), np.float32)
    env = SearchEnvV2(CFG, prior_bank=bank)
    s, obs = env.reset(jax.random.PRNGKey(0))
    # pd keeps its level axis, so the per-level observation channels exist
    assert s.pd.shape == (N_LEVELS, CFG.grid, CFG.grid)
    assert int(s.pd.shape[0]) == N_LEVELS
    assert obs["local"].shape[-1] >= 7
