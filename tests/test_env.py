"""Invariant tests for the grid env (plan section 13)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sar.config import EnvConfig
from sar.env.grid_env import ACTIVE, SearchEnv, action_tables
from sar.baselines import policies as P

CFG = EnvConfig(grid=32, extent_min=16, extent_max=30, p_delay=0.0)


@pytest.fixture(scope="module")
def env():
    return SearchEnv(CFG)


def rollout(env, key, policy, steps=80):
    s, _ = env.reset(key)
    out = []
    for t in range(steps):
        key, k = jax.random.split(key)
        a = policy(env, s, k)
        prev = s
        s, _, r, d = env.step(s, a)
        out.append((prev, a, s, float(r)))
        if bool(d):
            break
    return out


def test_segments_take_ring_moves():
    off, val, dirs = action_tables(4)
    assert off.shape == (33, 4, 2)
    for a in range(32):
        steps = np.diff(np.concatenate([np.zeros((1, 2), int), off[a]]), axis=0)
        assert (np.abs(steps).max(1) == 1).all()           # each move is one cell
        assert np.abs(off[a, -1]).max() == 4               # ends on the ring
    assert val[32].sum() == 1                              # hold looks once


def test_belief_update_matches_poisson(env):
    s, _ = env.reset(jax.random.PRNGKey(0))
    a = jnp.full((env.n,), env.hold)
    s2, _, r, _ = env.step(s, a)
    p = s.pos[0]
    expect = s.belief[p[0], p[1]] * (1 - s.pd)
    assert np.isclose(float(s2.belief[p[0], p[1]]), float(expect), rtol=1e-5)


def test_detections_do_not_change_belief():
    # belief after a step depends only on looks: same actions, different detection rng
    env = SearchEnv(EnvConfig(grid=32, extent_min=16, extent_max=30, p_delay=0.0, fp_max=0.5))
    s, _ = env.reset(jax.random.PRNGKey(3))
    a = P.greedy(env, s, jax.random.PRNGKey(0))
    s_a, *_ = env.step(s, a)
    s_b, *_ = env.step(s.replace(key=jax.random.PRNGKey(99)), a)
    looks_same = bool((s_a.looks == s_b.looks).all())
    if looks_same:
        assert np.allclose(np.asarray(s_a.belief), np.asarray(s_b.belief))


def test_reward_bound(env):
    # cumulative reward from any cell never exceeds its reward-map value
    for seed in range(3):
        traj = rollout(env, jax.random.PRNGKey(seed + 10), P.greedy)
        s0 = traj[0][0]
        total = sum(r for *_, r in traj)
        assert total <= 1.0 + 1e-5
        final = traj[-1][2]
        per_cell = (s0.reward_map - final.reward_map) / s0.total0
        assert float(per_cell.min()) >= -1e-7
        assert np.all(np.asarray(per_cell) <= np.asarray(s0.reward_map / s0.total0) + 1e-7)


def test_unmasked_actions_stay_in_margin(env):
    s, _ = env.reset(jax.random.PRNGKey(5))
    m = env.action_mask(s)
    dil = jax.lax.reduce_window(s.mask.astype(jnp.int32), 0, jax.lax.max, (3, 3), (1, 1), "SAME") > 0
    ends = env.paths(s.pos)[:, :, -1]
    for i in range(env.n):
        if int(s.status[i]) != ACTIVE:
            continue
        for a in range(env.hold):
            if bool(m[i, a]):
                e = ends[i, a]
                assert 0 <= int(e[0]) < env.g and 0 <= int(e[1]) < env.g
                assert bool(dil[e[0], e[1]])


@pytest.mark.parametrize("policy", [P.greedy, P.lawnmower, P.lookahead])
def test_separation(env, policy):
    for seed in range(3):
        for prev, a, s, _ in rollout(env, jax.random.PRNGKey(100 + seed), policy):
            act = np.asarray(s.status == ACTIVE) & np.asarray(prev.status == ACTIVE)
            seg = np.asarray(s.seg)
            idx = np.where(act)[0]
            for x in idx:
                for y in idx:
                    if x < y:
                        d = np.abs(seg[x] - seg[y]).max(-1)
                        assert d.min() > CFG.min_sep, (seed, x, y, d)


def test_fleet_change_replans(env):
    cfg = EnvConfig(grid=32, extent_min=16, extent_max=30, p_leave=1.0, p_join=1.0)
    e = SearchEnv(cfg)
    for prev, a, s, _ in rollout(e, jax.random.PRNGKey(7), P.lawnmower, 200):
        act = np.asarray(s.status == ACTIVE)
        if act.any():
            acts = np.asarray(P.lawnmower(e, s, jax.random.PRNGKey(0)))
            assert ((acts >= 0) & (acts < e.n_actions)).all()


def test_planner_beats_lawnmower_on_peak():
    cfg = EnvConfig(grid=32, extent_min=26, extent_max=30, train_prior_kinds=(3,))
    e = SearchEnv(cfg)
    def disc(pol):
        tot = 0.0
        for seed in range(6):
            traj = rollout(e, jax.random.PRNGKey(200 + seed), pol, 60)
            tot += sum(r * 0.97**t for t, (*_, r) in enumerate(traj))
        return tot
    assert disc(P.lookahead) > disc(P.lawnmower)
