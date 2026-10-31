"""The `mcts` search mode for the orchestrator on env v2.

  mcts       `policy.plan(state, drone=i, hand=True)` on top of `sar.search.v2_service`:
             the hand value search (hand prior plus greedy rollout), no network needed

The env is built at the mission grid size (32, 64 or 96), so no padding is
needed: the hand search is grid-agnostic.

Altitude: the orchestrator's look model and the sim detector cover one cell at
`MissionConfig.search_alt` (23 m, the v2 low level). The high level would look
at 3 x 3 cells, which neither of them models yet. So the state sets the high
level Pd to zero (no expected gain up there) and the separation mask removes
every high-level action. Every decision is therefore a low-level segment and
the returned altitude is `level_m[LOW]`.

Separation is the v1 rule: slot i's segment must stay more than `min_sep`
cells from every other active drone's current segment and cell. `mcts`
checks its pick afterwards and falls back to the greedy (hand prior) best legal action. If nothing but hold
is legal and the drone is too close to another, it escapes to the legal
segment whose end is farthest from the others.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sar.config_v2 import LOW, PD_TABLE_PATH, EnvConfigV2, load_pd_table
from sar.env.v2.grid_env_v2 import ACTIVE, AIRBORNE, LEFT, EnvStateV2, SearchEnvV2

MODES_V2 = ("mcts",)
ROOT = Path(__file__).resolve().parents[2]

_CACHE: dict = {}


def get_policy_v2(g: int, n_slots: int = 3, seed: int = 0) -> "SearchPolicyV2Modes":
    """One adapter per grid per process, so jit compiles are shared."""
    k = (int(g), int(n_slots))
    if k not in _CACHE:
        _CACHE[k] = SearchPolicyV2Modes(g, n_slots=n_slots, seed=seed)
    return _CACHE[k]


class SearchPolicyV2Modes:
    """Same `decide` inputs as `SearchPolicy`; returns (action, cells, alt_m)."""

    def __init__(self, g: int, n_slots: int = 3, seed: int = 0,
                 sims: int = 32, search_depth: int = 8):
        self.cfg = EnvConfigV2(grid=g, n_agents=n_slots).with_grid(g)
        self.g, self.n = g, n_slots
        self.seed, self.sims, self.depth = seed, sims, search_depth
        self._env = None
        self._pol = None
        self._warm = set()
        self._greedy = None
        self.timing: list = []      # (mode, seconds) per real decision

    # ------------------------------------------------------------- lazy setup
    @property
    def env(self) -> SearchEnvV2:
        if self._env is None:
            pdt = load_pd_table(ROOT / PD_TABLE_PATH)
            self._env = SearchEnvV2(self.cfg, pd_table=pdt)
            self.offsets = np.asarray(self._env.offsets)        # [2,A0,R,2]
            self.dirs = np.asarray(self._env.dirs)              # [2,A0,R]
            a0 = self._env.a0
            lev = np.arange(self._env.n_actions) // a0
            base = np.arange(self._env.n_actions) % a0
            self.paths_rel = self.offsets[lev, base]            # [66,R,2]
            self.lev_of = lev
            self.base_of = base
        return self._env

    def policy(self):
        """The hand search, built on first use, CPU only."""
        if self._pol is None:
            from sar.search import v2_service as S
            self._pol = S.SearchPolicyV2(None, None, {}, self.env, sims=self.sims,
                                         search_depth=self.depth, seed=self.seed)
            self._pol.device = jax.devices("cpu")[0]
        return self._pol

    def warmup(self, mode, state=None):
        """Compile the mode for every slot once (a compile in flight stalls setpoints)."""
        if mode in self._warm:
            return
        pol = self.policy()
        if state is None:
            state = self._sample_state()
        for i in range(self.n):
            pol.plan(state, drone=i, hand=True)
        self._greedy_fn()(state)
        self._warm.add(mode)

    def _sample_state(self):
        g = self.g
        mask = np.zeros((g, g), bool)
        mask[1:g - 1, 1:g - 1] = True
        b = mask / mask.sum()
        from sar.mission.search_policy import AgentView
        agents = [AgentView(active=True, cell=(g // 2, 2 + 6 * i)) for i in range(self.n)]
        return self.build_state(b, mask, np.zeros((g, g), np.int32), agents, (g // 2, g // 2),
                                0.8, 1.0, 0.0)

    def _greedy_fn(self):
        if self._greedy is None:
            from sar.search.mcts import hand_prior
            env = self.env
            # greedy fallback logits and the env's legal mask in one jitted call
            self._greedy = jax.jit(lambda s: (hand_prior(env, s)[0], env.action_mask(s)))
        return self._greedy

    # ----------------------------------------------------------------- state
    def build_state(self, belief, mask, looks, agents, home_cell, pd, belief0, t_frac):
        env, n, r, g = self.env, self.n, self.cfg.ring, self.g
        pos = np.zeros((n, 2), np.int32)
        seg = np.zeros((n, r, 2), np.int32)
        status = np.full(n, LEFT, np.int32)
        heading = np.full(n, -1, np.int32)
        battery = np.zeros(n, np.float32)
        freeze = np.zeros(n, np.int32)
        for i, a in enumerate(agents[:n]):
            pos[i] = np.clip(a.cell, 0, g - 1)
            s = list(a.seg)[-r:] or [a.cell]
            s = [s[0]] * (r - len(s)) + s
            seg[i] = np.clip(np.asarray(s), 0, g - 1)
            status[i] = ACTIVE if a.active else LEFT
            heading[i] = a.heading
            battery[i] = a.battery
            freeze[i] = int(a.confirming)
        pdm = np.zeros((2, g, g), np.float32)
        pdm[LOW] = float(pd)                 # the orchestrator's look model; high level unmodelled
        b = jnp.asarray(belief, jnp.float32)
        tot = jnp.float32(max(float(belief0), 1e-9))
        zeros = jnp.zeros((g, g), jnp.float32)
        dev = jax.devices("cpu")[0]
        st = EnvStateV2(
            belief=b, reward_map=b, mask=jnp.asarray(mask, bool), looks=jnp.asarray(looks, jnp.int32),
            pos=jnp.asarray(pos), heading=jnp.asarray(heading), status=jnp.asarray(status),
            level=jnp.full((n,), LOW, jnp.int32), battery=jnp.asarray(battery),
            sorties=jnp.zeros((n,), jnp.int32), ground=jnp.full((n,), AIRBORNE, jnp.int32),
            swap_left=jnp.zeros((n,), jnp.float32), freeze=jnp.asarray(freeze), seg=jnp.asarray(seg),
            join_step=jnp.zeros((n,), jnp.int32), leave_step=jnp.full((n,), 10**6, jnp.int32),
            home=jnp.asarray(np.clip(home_cell, 0, g - 1), jnp.int32),
            pd=jnp.asarray(pdm), fp=jnp.float32(0.0), floor_m=zeros, dem_m=zeros,
            total0=tot, belief0=tot,
            step=jnp.int32(int(min(max(t_frac, 0.0), 1.0) * (self.cfg.max_steps - 1))),
            key=jax.random.PRNGKey(0),
        )
        return jax.device_put(st, dev)

    def separation_mask(self, agents, i):
        """[66] bool for slot i: low level only, segments clear of the others' segments."""
        env = self.env
        ok = self.lev_of == LOW
        paths = np.asarray(agents[i].cell)[None, None] + self.paths_rel        # [66,R,2]
        for j, a in enumerate(agents[: self.n]):
            if j == i or not a.active:
                continue
            cells = np.asarray(list(a.seg) + [a.cell])
            d = np.abs(paths[:, :, None, :] - cells[None, None]).max(-1)
            ok &= ~(d <= self.cfg.min_sep).any((1, 2))
        ok[env.hold] = True
        return ok

    # ---------------------------------------------------------------- decide
    def decide(self, mode, i, belief, mask, looks, agents, home_cell, pd, belief0, t_frac):
        """Returns (action, segment cells [(r, c)], altitude AGL m) for slot i."""
        if mode not in MODES_V2:
            raise ValueError(f"unknown v2 mode {mode}")
        t0 = time.perf_counter()
        env = self.env
        state = self.build_state(belief, mask, looks, agents, home_cell, pd, belief0, t_frac)
        self.warmup(mode, state)
        pol = self.policy()
        extra = self.separation_mask(agents, i)
        a = int(pol.plan(state, drone=i, hand=True))
        if not extra[a]:
            lg, legal = self._greedy_fn()(state)
            ok = np.asarray(legal)[i] & extra
            a = int(np.argmax(np.where(ok, np.asarray(lg)[i], -np.inf)))
        if self.base_of[a] == env.a0 - 1:           # hold at either level
            a = self._escape(state, agents, i)
        self.timing.append((mode, time.perf_counter() - t0))
        alt = float(self.cfg.level_m[LOW])
        if self.base_of[a] == env.a0 - 1:
            return env.hold, [tuple(map(int, agents[i].cell))], alt
        cells = np.asarray(agents[i].cell) + self.paths_rel[a]
        return a, [tuple(map(int, c)) for c in cells], float(self.cfg.level_m[self.lev_of[a]])

    def _escape(self, state, agents, i):
        """Same rule as `SearchPolicy._escape`, on the low-level v2 segments."""
        env = self.env
        me = np.asarray(agents[i].cell)
        others = [np.asarray(list(a.seg) + [a.cell]) for j, a in enumerate(agents[: self.n])
                  if j != i and a.active]
        if not others:
            return env.hold
        cells = np.concatenate(others)
        if np.abs(cells - me).max(-1).min() > self.cfg.min_sep + 1:
            return env.hold
        g = self.g
        low = np.arange(env.a0)                       # low-level actions 0..a0-1
        ends = me[None] + self.paths_rel[low, -1]
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
        env = self.env
        if self.base_of[a] == env.a0 - 1:
            return -1
        return int(self.dirs[self.lev_of[a], self.base_of[a], -1])
