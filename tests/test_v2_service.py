"""Milestone 4: fleet search, expert iteration pieces and the inference API.

Uses a small untrained checkpoint written to a temp dir, so the tests do not
depend on a training run. The timing test uses the real checkpoint at grid 128
when `models/expert_iteration/final` exists and is skipped otherwise.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sar.config_v2 import TrainConfigV2
from sar.search import mcts, v2_service
from sar.train import expert_v2, networks_v2 as NN

FINAL = Path("models/expert_iteration/final")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    out = tmp_path_factory.mktemp("ckpt")
    net = NN.PolicyValue(width=16, embed=32, use_gain=True)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 31, 31, 7)),
                      jnp.zeros((1, 32, 32, 5)), jnp.zeros((1, 16)),
                      jnp.ones((1, 66), bool), jnp.ones((1, 66)))
    expert_v2.save_ckpt(out, params, {"width": 16, "embed": 32, "n_actions": 66,
                                      "use_gain": True,
                                      "value_scale": NN.VALUE_SCALE, "crop": 31})
    return out


@pytest.fixture(scope="module")
def policy(ckpt):
    return v2_service.load_policy(ckpt, grid=32, sims=4, search_depth=3)


def test_returns_to_go():
    r = np.array([[1.0, 0.0, 2.0]], np.float32)
    g = 0.5
    np.testing.assert_allclose(expert_v2.returns_to_go(r, g),
                               [[1.0 + 0.0 + 0.25 * 2.0, 1.0, 2.0]])


def test_act_returns_legal_actions(policy):
    env = policy.env
    for k in range(3):
        s, obs = env.reset(jax.random.PRNGKey(k))
        a = policy.act(policy.observe(s))
        assert a.shape == (env.n,)
        legal = np.asarray(obs["action_mask"])
        assert legal[np.arange(env.n), a].all()


def test_plan_fleet_and_single_drone(policy):
    env = policy.env
    s, obs = env.reset(jax.random.PRNGKey(3))
    a = policy.plan(s)
    legal = np.asarray(obs["action_mask"])
    assert a.shape == (env.n,) and legal[np.arange(env.n), a].all()
    a0 = policy.plan(s, drone=0)
    assert isinstance(a0, int) and legal[0, a0]
    ah = policy.plan(s, drone=1, hand=True)
    assert legal[1, ah]


def test_fleet_search_tracks_belief_in_tree():
    """A node's embedding carries its own belief, so repeated looks pay less."""
    env = expert_v2.make_env(32, p_delay=0.0)
    run = mcts.make_fleet_search_fn(env, TrainConfigV2(search_depth=3))
    s, _ = env.reset(jax.random.PRNGKey(1))
    res = jax.jit(lambda s, k: run(None, s, k, 8))(s, jax.random.PRNGKey(0))
    assert res.action.shape == (env.n,)
    assert np.allclose(np.asarray(res.visits).sum(-1), 8)
    w = np.asarray(res.weights)
    assert np.allclose(w.sum(-1), 1.0, atol=1e-4)
    legal = np.asarray(env.action_mask(s))
    assert (w[~legal] < 1e-6).all()


def test_gen_and_train_round_trip(tmp_path):
    """Planner data -> a short training run -> a loadable checkpoint."""
    import subprocess
    import sys
    d, c = tmp_path / "d", tmp_path / "c"
    base = [sys.executable, "-m", "sar.train.expert_v2"]
    subprocess.run(base + ["gen", "--policy", "greedy", "--grid", "32", "--episodes", "2",
                           "--batch", "2", "--keep-steps", "16", "--out", str(d)], check=True)
    z = np.load(sorted(d.glob("shard_*.npz"))[0])
    assert z["local"].dtype == np.float16 and (z["ret"] >= 0).all()
    assert np.allclose(z["target"].astype(np.float32).sum(-1), 1.0, atol=1e-2)
    subprocess.run(base + ["train", "--data", str(d), "--out", str(c), "--iters", "4",
                           "--batch", "16", "--eval-every", "2", "--width", "16",
                           "--embed", "32", "--gain"], check=True)
    net, params, meta = expert_v2.load_ckpt(c)
    assert meta["iters"] == 4 and net.use_gain


@pytest.mark.skipif(not FINAL.exists(), reason="no trained checkpoint")
def test_cpu_decision_under_200ms():
    pol = v2_service.load_policy(FINAL, grid=128)
    s, obs = pol.env.reset(jax.random.PRNGKey(0))
    pol.warmup(s, drones=(0,))
    t = time.perf_counter()
    for _ in range(3):
        pol.act(pol.observe(s))
    t_act = (time.perf_counter() - t) / 3
    t = time.perf_counter()
    for _ in range(3):
        pol.plan(s, drone=0)
    t_plan = (time.perf_counter() - t) / 3
    t = time.perf_counter()
    pol.plan(s, drone=0, hand=True)
    for _ in range(3):
        pol.plan(s, drone=0, hand=True)
    t_hand = (time.perf_counter() - t) / 4
    assert t_act < 0.2, t_act
    assert t_plan < 0.2, t_plan
    assert t_hand < 2.0, t_hand        # includes one compile
