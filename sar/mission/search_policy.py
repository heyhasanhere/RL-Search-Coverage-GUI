"""Search policy inference on CPU for the orchestrator (plan sections 4.1, 5.5).

The orchestrator keeps its own world model. For each decision it builds a
JAX EnvState from that model and runs the chosen method (RL, greedy or
lawnmower) for one drone, using the predicted position at the end of the
drone's current segment. Segments that pass near another drone's current
segment are masked (plan 4.6).
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from sar.baselines.policies import POLICIES
from sar.config import EnvConfig
from sar.env.grid_env import ACTIVE, LEFT, EnvState, SearchEnv
from sar.mission.search_policy_v2 import MODES_V2
from sar.train.networks import Actor

V1_MODES = ("rl", "greedy", "lawnmower")
# rl first: the UI and start_mission default to the trained RL policy; mcts runs in search_policy_v2
MODES = (*V1_MODES, *MODES_V2)
GRIDS = (32, 64, 96)


def grid_for_extent(extent_cells: int) -> int:
    for g in GRIDS:
        if extent_cells <= g - 2:
            return g
    raise ValueError(f"area extent {extent_cells} cells exceeds the largest grid")


@dataclass
class AgentView:
    """One search slot as the policy sees it."""
    active: bool
    cell: tuple            # current (or predicted) cell
    heading: int = -1
    battery: float = 1500.0
    confirming: bool = False
    seg: tuple = ()        # current segment cells, list of (r, c)


class SearchPolicy:
    def __init__(self, g: int, ckpt_root: Path | None, n_slots: int = 3, seed: int = 0, cfg_kw=None):
        self.cfg = EnvConfig(grid=g, n_agents=n_slots, **(cfg_kw or {}))
        self.env = SearchEnv(self.cfg)
        self.g, self.n = g, n_slots
        self.key = jax.random.PRNGKey(seed)
        self.ckpt_root = ckpt_root
        self._fns = {}
        self._actor = None

    # --------------------------------------------------------------- methods
    def _rl_params(self):
        if self._actor is None:
            path = Path(self.ckpt_root) / f"g{self.g}_ppo_actor.msgpack"
            actor = Actor(self.env.n_actions)
            _, obs = self.env.reset(jax.random.PRNGKey(0))
            tmpl = actor.init(jax.random.PRNGKey(0), obs["local"], obs["global"], obs["scalars"], obs["action_mask"])
            self._actor = (actor, serialization.from_bytes(tmpl, path.read_bytes()))
        return self._actor

    def _scores_fn(self, mode):
        """Jitted (state, key, extra_mask[N,A]) -> action per slot."""
        if mode in self._fns:
            return self._fns[mode]
        env = self.env
        if mode == "rl":
            actor, params = self._rl_params()

            def fn(state, key, extra):
                obs = env.observe(state)
                lg = actor.apply(params, obs["local"], obs["global"], obs["scalars"], obs["action_mask"] & extra)
                return jnp.argmax(lg, -1)
        else:
            base = POLICIES[mode]

            def fn(state, key, extra):
                a = base(env, state, key)
                ok = extra[jnp.arange(env.n), a]
                return jnp.where(ok, a, env.hold)
        self._fns[mode] = jax.jit(fn)
        return self._fns[mode]

    # ----------------------------------------------------------------- state
    def build_state(self, belief, mask, looks, agents: list[AgentView], home_cell, pd, belief0, t_frac):
        n, r = self.n, self.cfg.ring
        pos = np.zeros((n, 2), np.int32)
        seg = np.zeros((n, r, 2), np.int32)
        status = np.full(n, LEFT, np.int32)
        heading = np.full(n, -1, np.int32)
        battery = np.zeros(n, np.float32)
        freeze = np.zeros(n, np.int32)
        for i, a in enumerate(agents):
            pos[i] = np.clip(a.cell, 0, self.g - 1)
            s = list(a.seg)[-r:] or [a.cell]
            s = [s[0]] * (r - len(s)) + s
            seg[i] = np.clip(np.asarray(s), 0, self.g - 1)
            status[i] = ACTIVE if a.active else LEFT
            heading[i] = a.heading
            battery[i] = a.battery
            freeze[i] = int(a.confirming)
        b = jnp.asarray(belief, jnp.float32)
        return EnvState(
            belief=b, reward_map=b, mask=jnp.asarray(mask), looks=jnp.asarray(looks, jnp.int32),
            pos=jnp.asarray(pos), heading=jnp.asarray(heading), status=jnp.asarray(status),
            battery=jnp.asarray(battery), freeze=jnp.asarray(freeze), seg=jnp.asarray(seg),
            join_step=jnp.zeros(n, jnp.int32), leave_step=jnp.full(n, 10**6, jnp.int32),
            home=jnp.asarray(np.clip(home_cell, 0, self.g - 1), jnp.int32),
            pd=jnp.float32(pd), fp=jnp.float32(0.0),
            total0=jnp.float32(max(belief0, 1e-9)), belief0=jnp.float32(max(belief0, 1e-9)),
            step=jnp.int32(int(t_frac * self.cfg.max_steps)), key=jax.random.PRNGKey(0),
        )

    def separation_mask(self, agents: list[AgentView], i: int):
        """[N,A] mask: slot i's segments must stay min_sep away from others' segments."""
        env = self.env
        extra = np.ones((self.n, env.n_actions), bool)
        me = np.asarray(agents[i].cell)
        paths = me[None, None] + np.asarray(env.offsets)              # [A,R,2]
        for j, a in enumerate(agents):
            if j == i or not a.active:
                continue
            cells = np.asarray(list(a.seg) + [a.cell])                 # [K,2]
            d = np.abs(paths[:, :, None, :] - cells[None, None]).max(-1)  # [A,R,K]
            extra[i] &= ~(d <= self.cfg.min_sep).any((1, 2))
        extra[i, env.hold] = True
        return jnp.asarray(extra)

    def decide(self, mode, i, belief, mask, looks, agents, home_cell, pd, belief0, t_frac):
        """Returns (action, segment cells [R,(r,c)]) for slot i."""
        if mode not in V1_MODES:
            raise ValueError(f"unknown mode {mode}")
        state = self.build_state(belief, mask, looks, agents, home_cell, pd, belief0, t_frac)
        self.key, k = jax.random.split(self.key)
        acts = self._scores_fn(mode)(state, k, self.separation_mask(agents, i))
        a = int(acts[i])
        if a == self.env.hold:
            a = self._escape(state, agents, i)
        if a == self.env.hold:
            return a, [tuple(map(int, agents[i].cell))]
        cells = np.asarray(agents[i].cell) + np.asarray(self.env.offsets[a])
        return a, [tuple(map(int, c)) for c in cells]

    def _escape(self, state, agents, i):
        """Two drones closer than min_sep can mask each other's every move.
        Then move to the legal segment whose endpoint is farthest from the others."""
        env = self.env
        me = np.asarray(agents[i].cell)
        others = [np.asarray(list(a.seg) + [a.cell]) for j, a in enumerate(agents) if j != i and a.active]
        if not others:
            return env.hold
        cells = np.concatenate(others)
        if np.abs(cells - me).max(-1).min() > self.cfg.min_sep + 1:
            return env.hold
        # polygon legality only: the env mask also blocks every move near another drone
        ends = me[None] + np.asarray(env.offsets)[:, -1]
        g = self.g
        m = np.asarray(state.mask)
        dil = m.copy()
        dil[1:] |= m[:-1]; dil[:-1] |= m[1:]
        dil[:, 1:] |= dil[:, :-1].copy(); dil[:, :-1] |= dil[:, 1:].copy()
        inb = ((ends >= 0) & (ends < g)).all(-1)
        ec = np.clip(ends, 0, g - 1)
        legal = inb & dil[ec[:, 0], ec[:, 1]]
        dmin = np.abs(ends[:, None, :] - cells[None]).max(-1).min(-1)
        dmin = np.where(legal, dmin, -1)
        dmin[env.hold] = -1
        a = int(np.argmax(dmin))
        return a if dmin[a] > 0 else env.hold

    def heading_of(self, a):
        if a == self.env.hold:
            return -1
        return int(self.env.dirs[a, -1])
