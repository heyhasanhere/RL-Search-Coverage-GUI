"""MCTS planner tests (plan section 19).

The two properties the plan asks for: a search with one simulation reproduces
the greedy baseline exactly, and raising the simulation count never lowers the
expected return on fixed seeds.
"""

import jax
import numpy as np
import pytest

from sar.config_v2 import EnvConfigV2, TrainConfigV2
from sar.env.v2.grid_env_v2 import SearchEnvV2
from sar.search import mcts, policies_v2 as P

CFG = EnvConfigV2(grid=32, extent_min=26, extent_max=30, p_delay=0.0).with_grid(32)
TCFG = TrainConfigV2(search_depth=4, rollout_depth=3)


@pytest.fixture(scope="module")
def env():
    return SearchEnvV2(CFG)


@pytest.fixture(scope="module")
def run(env):
    return mcts.make_search_fn(env, TCFG)


def test_one_simulation_equals_greedy(env, run):
    """With a single simulation the tree can only expand the top prior."""
    agree = total = 0
    for k in range(8):
        s, _ = env.reset(jax.random.PRNGKey(k))
        a_search = np.asarray(run(None, s, jax.random.PRNGKey(k), 1).action)
        a_greedy = np.asarray(P.greedy(env, s, jax.random.PRNGKey(k)))
        agree += int((a_search == a_greedy).sum())
        total += len(a_greedy)
    assert agree == total, (agree, total)


def test_actions_are_legal_and_visits_sum_to_simulations(env, run):
    s, _ = env.reset(jax.random.PRNGKey(0))
    legal = np.asarray(env.action_mask(s))
    for sims in (1, 4, 16):
        r = run(None, s, jax.random.PRNGKey(sims), sims)
        a = np.asarray(r.action)
        assert legal[np.arange(env.n), a].all(), (sims, a)
        assert np.asarray(r.visits).sum(-1).tolist() == [sims] * env.n


def test_more_simulations_do_not_lower_expected_value(env, run):
    """The root value estimate must not fall as simulations are added."""
    vals = {}
    for sims in (2, 8, 32):
        per = []
        for k in range(6):
            s, _ = env.reset(jax.random.PRNGKey(100 + k))
            per.append(float(np.mean(np.asarray(run(None, s, jax.random.PRNGKey(k), sims).value))))
        vals[sims] = float(np.mean(per))
    assert vals[32] >= vals[8] - 1e-6, vals
    assert vals[8] >= vals[2] - 1e-6, vals


def test_search_beats_random_on_return(env, run):
    """MCTS should not be worse than random legal actions over short episodes."""
    gamma = 0.995

    def episode(policy, key, steps=60):
        s, _ = env.reset(key)
        disc, x = 0.0, 1.0
        for t in range(steps):
            a = policy(s, jax.random.fold_in(key, t))
            s, _, r, d = env.step(s, a)
            disc += x * float(r)
            x *= gamma
            if d:
                break
        return disc

    keys = [jax.random.PRNGKey(7 + i) for i in range(4)]
    mcts_ret = np.mean([episode(lambda s, k: run(None, s, k, 8).action, kk) for kk in keys])
    rand_ret = np.mean([episode(
        lambda s, k: jax.random.randint(k, (env.n,), 0, env.n_actions), kk) for kk in keys])
    assert mcts_ret >= rand_ret, (mcts_ret, rand_ret)


def test_visit_counts_sum_and_are_teacher_usable(env, run):
    """Visits are the per-action teacher target for expert iteration."""
    s, _ = env.reset(jax.random.PRNGKey(1))
    r = run(None, s, jax.random.PRNGKey(0), 16)
    v = np.asarray(r.visits)
    assert v.shape == (env.n, env.n_actions)
    assert (v >= 0).all() and v.sum(-1).tolist() == [16] * env.n
    # The action taken is a legal, searched branch, and the action weights the
    # planner returns form the target distribution M4 regresses onto. mctx picks
    # the executed action by sampling `action_weights`, so it need not be the
    # single most-visited child; both only have to be legal and supported.
    legal = np.asarray(env.action_mask(s))
    assert legal[np.arange(env.n), np.asarray(r.action)].all()
    w = np.asarray(r.weights)
    assert w.shape == (env.n, env.n_actions)
    assert np.allclose(w.sum(-1), 1.0, atol=1e-5)
    assert (w[~legal] == 0.0).all()