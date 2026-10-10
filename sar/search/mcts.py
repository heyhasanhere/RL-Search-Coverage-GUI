"""MCTS planner for env v2, built on DeepMind `mctx` (Gumbel MuZero).

Design (plan section 8.2):

* The model is the true environment, so one simulation is one real env step.
* The mctx batch axis is the fleet: every drone runs its own tree row. The
  shared world (belief, mask, Pd maps, floor, home) is static over the depth of
  one search and is captured once; the embedding carries only the per-drone
  mutable fields, every leaf of which has the batch axis (the drone axis)
  first, as mctx requires.
* The deciding drone at each node searches all 66 actions. The other drones
  follow their committed segments and then a default policy, so a simulation
  stays a legal fleet move without searching N * 66 branches.
* Prior and value come from a hand function of the state (immediate expected
  detections plus a pull potential) and a greedy rollout. Both are pure and only
  reached through `hand_prior` / `greedy_rollout`, so milestone 4 can replace
  them with a trained network by passing `params` into the recurrent function
  without touching the search loop.

Entry point: `search(params_or_None, state, key, num_simulations)`.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx

from sar.config_v2 import TrainConfigV2
from sar.env.v2.grid_env_v2 import SearchEnvV2
from sar.search import policies_v2

NEG = -1e9

# Per-drone state fields that change as a search simulates. Every one of these
# has the drone axis first, so the pytree batches cleanly under mctx.
DRONE_FIELDS = ("pos", "heading", "status", "level", "battery", "sorties",
                "ground", "swap_left", "freeze", "seg")


class Embed(NamedTuple):
    """Search embedding: per-drone mutable fields, drone axis first."""

    pos: jax.Array
    heading: jax.Array
    status: jax.Array
    level: jax.Array
    battery: jax.Array
    sorties: jax.Array
    ground: jax.Array
    swap_left: jax.Array
    freeze: jax.Array
    seg: jax.Array


@dataclass
class SearchResult:
    """Result of one search call.

    action   [N] int32  chosen action per drone
    visits   [N, 66]    root child visit counts (the teacher target)
    value    [N]        root value estimate
    weights  [N, 66]    visit-proportional action weights
    qvalues  [N, 66]    root action q-values (diagnostics)
    tree     mctx.Tree  the raw search tree, for teacher-data extraction
    """

    action: jax.Array
    visits: jax.Array
    value: jax.Array
    weights: jax.Array
    qvalues: jax.Array
    tree: object


# Register as a pytree so a jitted `run` can return it. The raw tree is a child
# (it is a dict of arrays), which lets callers that only need the arrays slice
# them off without re-running the search.
jax.tree_util.register_pytree_node(
    SearchResult,
    lambda r: ((r.action, r.visits, r.value, r.weights, r.qvalues, r.tree), None),
    lambda _aux, children: SearchResult(*children),
)


def embed(state) -> Embed:
    return Embed(*(getattr(state, f) for f in DRONE_FIELDS))


def static_world(state):
    """The state fields that are shared across drones and static in a search."""
    return {f: getattr(state, f) for f in state.__dataclass_fields__
            if f not in DRONE_FIELDS}


def reassemble(emb: Embed, world):
    """Build a full env state from an embedding and the static world."""
    from sar.env.v2.grid_env_v2 import EnvStateV2
    return EnvStateV2(**{**world, **emb._asdict()})


def hand_prior(env: SearchEnvV2, state):
    """Prior logits [N, 66] and greedy value [N] of a state.

    The prior is the log of the immediate expected-detection score of each
    action plus a small pull toward the belief still out there, so Gumbel noise
    explores over promising moves. The value is that score normalised by the
    episode's total belief mass so values stay O(1) whatever the map size,
    which is what mctx's qtransform expects.
    """
    n, A = env.n, env.n_actions
    legal = env.action_mask(state)
    acts = jnp.broadcast_to(jnp.arange(A)[None, :], (n, A))
    gain = policies_v2.seg_value(env, state.belief, state.pd, state.pos, acts)
    ends = policies_v2._ends(env, state.pos, acts)
    phi = policies_v2.potential(env, state.belief, state.mask, ends)
    scale = 1.0 / jnp.maximum(state.total0, 1e-6)
    # Same score the greedy baseline maximises, so that a search with a single
    # simulation (which can only expand one child) reproduces greedy exactly.
    score = (gain + 1e-3 * phi / env.r) * scale
    logits = jnp.where(legal, jnp.log(jnp.maximum(score, 1e-8)), NEG)
    value = jnp.max(jnp.where(legal, score, NEG), -1)
    return logits, value


def greedy_rollout(env: SearchEnvV2, state, depth: int, gamma: float, key):
    """Greedy rollout of `depth` steps; returns the discounted reward [N]."""
    tot = jnp.zeros((env.n,), jnp.float32)
    disc = jnp.ones((env.n,), jnp.float32)
    s, k = state, key
    for _ in range(depth):
        s, _, r, d = env.step(s, policies_v2.greedy(env, s, k))
        tot = tot + disc * r
        disc = disc * gamma * jnp.where(d, 0.0, 1.0)
        k, sk = jax.random.split(k)
        s = s.replace(key=sk)
    return tot


def make_search_fn(env: SearchEnvV2, tcfg: TrainConfigV2):
    """Build the jittable `run(params, state, key, num_simulations)`."""

    gamma, depth, rollout = tcfg.gamma, tcfg.search_depth, tcfg.rollout_depth

    def root_fn(state):
        logits, value = hand_prior(env, state)
        return mctx.RootFnOutput(prior_logits=logits, value=value,
                                 embedding=embed(state))

    def make_recurrent(world):
        def recurrent_fn(params, key, action, emb):
            state = reassemble(emb, world)
            k1, k2 = jax.random.split(key)
            # the deciding action comes from this node's row; the rest of the
            # fleet follows its committed segment then the default policy
            default = policies_v2.greedy(env, state, k1)
            acts = jnp.where(state.status == 1, action, default)
            s_next, _, reward, done = env.step(state, acts)
            _, base = hand_prior(env, s_next)
            tail = greedy_rollout(env, s_next, rollout, gamma, k2)
            # env reward and done are whole-fleet scalars; mctx wants one value
            # per tree row, so both are shared across the fleet. The reward is a
            # cooperative team reward, which is the quantity we actually optimise.
            r = jnp.asarray(reward) * jnp.ones((env.n,), jnp.float32)
            alive = jnp.where(done, 0.0, 1.0) * jnp.ones((env.n,), jnp.float32)
            value = (base + tail) * alive
            logits, _ = hand_prior(env, s_next)
            return (mctx.RecurrentFnOutput(
                reward=r * alive,
                discount=jnp.full((env.n,), gamma) * alive,
                prior_logits=logits,
                value=value,
            ), embed(s_next))
        return recurrent_fn

    def run(params, state, key, num_simulations):
        world = static_world(state)
        recurrent_fn = make_recurrent(world)
        out = mctx.gumbel_muzero_policy(
            params,
            key,
            root_fn(state),
            recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~env.action_mask(state),
            max_depth=depth,
            max_num_considered_actions=tcfg.max_considered,
            # With a single simulation the tree can only expand one child, so
            # Gumbel noise would pick a random among the top priors instead of
            # the best one. Turning it off makes one simulation exactly greedy.
            gumbel_scale=jnp.where(num_simulations <= 1, 0.0, 1.0),
        )
        tree = out.search_tree
        n_nodes = tree.node_visits.shape[1]
        root_idx = jnp.zeros((env.n, n_nodes), jnp.int32)
        return SearchResult(
            action=out.action,
            visits=tree.children_visits[:, 0, :],
            value=tree.node_values[:, 0],
            weights=out.action_weights,
            qvalues=tree.qvalues(root_idx)[:, 0, :],
            tree=tree,
        )

    return run


def search(params, env: SearchEnvV2, state, key, num_simulations: int,
           tcfg: TrainConfigV2 | None = None, run=None):
    """Search for one fleet move.

    `params` may be None (hand prior and value) or a network's parameters, in
    which case the call signature is kept for milestone 4. Pass `run` (built by
    `make_search_fn`) to reuse the traced search across steps.
    """
    run = run or make_search_fn(env, tcfg or TrainConfigV2())
    return run(params, state, key, num_simulations)

# --------------------------------------------------------------------------
# Fleet search (milestone 4).
#
# `make_search_fn` above keeps the belief and reward map in the static world,
# so a node deeper in the tree is scored on the root belief: a path that looks
# at the same cells twice is credited twice. It also builds each simulated
# fleet state from the drone fields of different tree rows, which may sit at
# different nodes. The fleet search below fixes both:
#
# * every tree row is one deciding drone and carries a full fleet state; the
#   belief, reward map and look counts are part of the node embedding, so the
#   model inside the tree is the true env;
# * at a node, the deciding drone takes the tree action and the other drones
#   take a default action stored with the node (greedy for the hand version,
#   the network's best action for the learned version, plan section 8.3).
#
# Prior and value come from `net` (a `networks_v2.PolicyValue`) when it is
# given, otherwise from `hand_prior` plus an optional greedy rollout.

MUTABLE_FIELDS = ("belief", "reward_map", "looks", "pos", "heading", "status",
                  "level", "battery", "sorties", "ground", "swap_left", "freeze",
                  "seg", "step", "key")


def _split_state(state):
    mut = {f: getattr(state, f) for f in MUTABLE_FIELDS}
    world = {f: getattr(state, f) for f in state.__dataclass_fields__
             if f not in MUTABLE_FIELDS}
    return mut, world


def make_fleet_search_fn(env: SearchEnvV2, tcfg: TrainConfigV2, net=None,
                         rows=None, rollout_depth: int = 0,
                         value_scale: float = 10.0):
    """Build `run(params, state, key, num_simulations, gumbel_scale=1.0)`.

    `rows` is the tuple of deciding drones (default: every drone). The result
    arrays have one row per entry of `rows`. `num_simulations` must be a Python
    int; wrap `run` in `jax.jit(..., static_argnums=3)` to reuse it.
    """
    from sar.env.v2.grid_env_v2 import EnvStateV2

    n, gamma = env.n, tcfg.gamma
    rows = tuple(range(n)) if rows is None else tuple(int(r) for r in rows)
    rows_arr = jnp.asarray(rows, jnp.int32)

    def evaluate(params, state):
        """Per-drone prior logits [N, A], value [N] and default actions [N]."""
        legal = env.action_mask(state)
        if net is None:
            logits, value = hand_prior(env, state)
        else:
            from sar.train.networks_v2 import apply_obs
            obs = env.observe(state)
            if net.use_gain:
                obs["gain"] = policies_v2.action_gain(env, state)
            logits, v = apply_obs(net, params, obs)
            logits = logits.astype(jnp.float32)
            # Every drone predicts the same team return; average the active ones.
            act = (state.status == 1).astype(jnp.float32)
            vt = (v * act).sum() / jnp.maximum(act.sum(), 1.0) / value_scale
            value = jnp.broadcast_to(vt, (n,))
        default = jnp.argmax(jnp.where(legal, logits, NEG), -1).astype(jnp.int32)
        return logits, value, default

    def run(params, state, key, num_simulations, gumbel_scale=1.0):
        mut0, world = _split_state(state)
        logits, value, default = evaluate(params, state)
        R = len(rows)
        emb0 = {k: jnp.broadcast_to(v, (R,) + v.shape) for k, v in mut0.items()}
        emb0["default"] = jnp.broadcast_to(default, (R, n))
        root = mctx.RootFnOutput(prior_logits=logits[rows_arr],
                                 value=value[rows_arr], embedding=emb0)

        def one_row(params, i, a, emb, k):
            k1, k2 = jax.random.split(k)
            mut = {f: emb[f] for f in MUTABLE_FIELDS}
            mut["key"] = k1
            s = EnvStateV2(**{**world, **mut})
            acts = emb["default"].at[i].set(a)
            s2, _, r, d = env.step(s, acts)
            lg, val, dflt = evaluate(params, s2)
            v = val[i]
            if rollout_depth > 0:
                v = v + greedy_rollout(env, s2, rollout_depth, gamma, k2)[i]
            alive = jnp.where(d, 0.0, 1.0)
            e2 = {f: getattr(s2, f) for f in MUTABLE_FIELDS}
            e2["default"] = dflt
            return r * 1.0, alive, lg[i], v * alive, e2

        def recurrent_fn(params, key, action, emb):
            ks = jax.random.split(key, R)
            r, alive, lg, v, e2 = jax.vmap(one_row, (None, 0, 0, 0, 0))(
                params, rows_arr, action, emb, ks)
            return mctx.RecurrentFnOutput(
                reward=r.astype(jnp.float32), discount=gamma * alive,
                prior_logits=lg, value=v.astype(jnp.float32)), e2

        legal = env.action_mask(state)[rows_arr]
        out = mctx.gumbel_muzero_policy(
            params, key, root, recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~legal,
            max_depth=tcfg.search_depth,
            max_num_considered_actions=tcfg.max_considered,
            gumbel_scale=jnp.where(num_simulations <= 1, 0.0, gumbel_scale),
        )
        tree = out.search_tree
        root_idx = jnp.zeros((R,), jnp.int32)
        return SearchResult(
            action=out.action.astype(jnp.int32),
            visits=tree.children_visits[:, 0, :],
            value=tree.node_values[:, 0],
            weights=out.action_weights,
            qvalues=tree.qvalues(root_idx),
            tree=None,
        )

    return run
