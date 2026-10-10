"""Inference API for the orchestrator: learned search policies on env v2.

    policy = load_policy("models/v2_m4/final")          # CPU by default
    actions = policy.act(policy.observe(state))            # 'rl_search', no search
    actions = policy.plan(state)                           # 'mcts', network-guided
    a_i = policy.plan(state, drone=i)                      # one drone's decision
    a_i = policy.plan(state, drone=i, hand=True)           # hand value search

`obs` is the dict `SearchEnvV2.observe` returns plus `gain`
(`policies_v2.action_gain`); `policy.observe(state)` builds it. `state` is an `EnvStateV2` of the env the
policy was built for (`policy.env`, or pass `env=` to `load_policy`). Both
return int actions in 0..65 (see `EnvConfigV2.n_actions`).

The network runs in float32 so CPU inference does not hit slow bfloat16
convolutions. Every entry point is jitted once per shape; call `warmup()` before
the mission so the first real decision does not pay the compile.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sar.config_v2 import TrainConfigV2
from sar.search import mcts, policies_v2 as P
from sar.train import networks_v2 as NN

DEFAULT_CKPT = Path("models/v2_m4/final")


class SearchPolicyV2:
    """A trained `PolicyValue` network with a no-search and a search mode."""

    def __init__(self, net, params, meta, env, sims=32, search_depth=8,
                 considered=16, gumbel_scale=0.0, seed=0, hand_rollout=4):
        self.hand_rollout = int(hand_rollout)
        self.net, self.params, self.meta, self.env = net, params, meta, env
        self.sims = int(sims)
        self.gumbel_scale = float(gumbel_scale)
        self.tcfg = TrainConfigV2(search_depth=search_depth, max_considered=considered)
        self._key = jax.random.PRNGKey(seed)
        self._act = jax.jit(self._act_impl)
        self._observe = jax.jit(self._observe_impl)
        self._plans = {}

    # ------------------------------------------------------------ rl_search
    def _act_impl(self, params, obs):
        logits, value = NN.apply_obs(self.net, params, obs)
        return (jnp.argmax(logits, -1).astype(jnp.int32),
                value / self.meta.get("value_scale", NN.VALUE_SCALE))

    def observe(self, state):
        """Observation dict for `act`: `env.observe` plus the `gain` feature."""
        dev = getattr(self, "device", None)
        if dev is not None:
            state = jax.device_put(state, dev)
        return self._observe(state)

    def _observe_impl(self, state):
        obs = self.env.observe(state)
        obs["gain"] = P.action_gain(self.env, state)
        return obs

    def act(self, obs, return_value=False):
        """Network-only actions [N] from an observation dict.

        A checkpoint trained with the gain feature needs `obs["gain"]`; build the
        dict with `observe(state)` (or add `policies_v2.action_gain`).
        """
        x = {"local": jnp.asarray(obs["local"], jnp.float32),
             "global": jnp.asarray(obs["global"], jnp.float32),
             "scalars": jnp.asarray(obs["scalars"], jnp.float32),
             "action_mask": jnp.asarray(obs["action_mask"], bool)}
        if self.net.use_gain:
            if "gain" not in obs:
                raise ValueError("this checkpoint needs obs['gain']; use policy.observe(state)")
            x["gain"] = jnp.asarray(obs["gain"], jnp.float32)
        a, v = self._act(self.params, x)
        a = np.asarray(a)
        return (a, np.asarray(v)) if return_value else a

    # ----------------------------------------------------------------- mcts
    def _plan_fn(self, drone, hand):
        k = (drone, hand)
        if k not in self._plans:
            rows = None if drone is None else (int(drone),)
            if hand:
                run = mcts.make_fleet_search_fn(self.env, self.tcfg, net=None, rows=rows,
                                                rollout_depth=self.hand_rollout)
            else:
                run = mcts.make_fleet_search_fn(
                    self.env, self.tcfg, net=self.net, rows=rows,
                    value_scale=self.meta.get("value_scale", NN.VALUE_SCALE))
            sims, gs = self.sims, self.gumbel_scale
            self._plans[k] = jax.jit(lambda p, s, key: run(p, s, key, sims, gs).action)
        return self._plans[k]

    def plan(self, state, drone=None, key=None, hand=False):
        """Fleet search on the true env model. Actions [N], or one int for `drone`.

        `hand=False` guides the search with the network (prior and value);
        `hand=True` uses the hand prior and a greedy rollout value instead, which
        scored higher on Tier 0 in M4 (docs/EXECUTION_LOG.md).
        """
        if key is None:
            self._key, key = jax.random.split(self._key)
        dev = getattr(self, "device", None)
        if dev is not None:
            state, key = jax.device_put((state, key), dev)
        params = None if hand else self.params
        a = np.asarray(self._plan_fn(drone, hand)(params, state, key))
        return int(a[0]) if drone is not None else a

    def warmup(self, state, drones=(None,), hand=False):
        """Compile `act` and `plan` for the given drones on a sample state."""
        self.act(self.observe(state))
        for d in drones:
            self.plan(state, drone=d, hand=hand)


def load_policy(ckpt=DEFAULT_CKPT, env=None, grid=128, n_agents=3, sims=32,
                search_depth=8, considered=16, gumbel_scale=0.0, cpu=True, seed=0):
    """Load a checkpoint directory (`params.msgpack`, `meta.json`).

    `cpu=True` pins the computation to the host CPU, which is what the
    orchestrator uses so the GPU stays free for the detector.
    """
    from sar.train.expert_v2 import load_ckpt, make_env

    if cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    dev = jax.devices("cpu")[0] if cpu else jax.devices()[0]
    with jax.default_device(dev):
        net, params, meta = load_ckpt(ckpt, dtype=jnp.float32)
        params = jax.device_put(params, dev)
        if env is None:
            env = make_env(grid, n_agents=n_agents)
        pol = SearchPolicyV2(net, params, meta, env, sims=sims, search_depth=search_depth,
                             considered=considered, gumbel_scale=gumbel_scale, seed=seed)
    pol.device = dev
    return pol
