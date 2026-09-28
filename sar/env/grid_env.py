"""GPU-vectorized multi-drone search grid environment (plan section 5).

One env instance is pure JAX and is batched with jax.vmap.
Agents decide in sync; each action is a 4-move segment to a ring cell or hold.
"""


import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from sar.config import EnvConfig
from sar.env import maps

# Agent status values
PENDING, ACTIVE, LEFT = 0, 1, 2

# 8 headings (dr, dc) indexed 0..7 counter-clockwise from east
_DIRS = np.array([(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)])


def action_tables(ring: int):
    """Ring endpoints, per-move offsets, validity, and move directions.

    Returns numpy arrays:
      offsets [A, ring, 2] cumulative cell offsets along the segment
      valid   [A, ring] bool, which offsets are looked at
      dirs    [A, ring] int heading index of each move (-1 for hold)
    The last action (index A-1) is hold: one look at the current cell.
    """
    ends = []
    for dr in range(-ring, ring + 1):
        for dc in range(-ring, ring + 1):
            if max(abs(dr), abs(dc)) == ring:
                ends.append((dr, dc))
    ends.sort(key=lambda e: np.arctan2(-e[0], e[1]) % (2 * np.pi))
    n = len(ends) + 1
    offsets = np.zeros((n, ring, 2), np.int32)
    valid = np.zeros((n, ring), bool)
    dirs = -np.ones((n, ring), np.int32)
    for a, (dr, dc) in enumerate(ends):
        prev = np.zeros(2, np.int32)
        for k in range(1, ring + 1):
            cur = np.floor(np.array([dr, dc]) * k / ring + 0.5).astype(np.int32)
            offsets[a, k - 1] = cur
            valid[a, k - 1] = True
            step = cur - prev
            dirs[a, k - 1] = int(np.where((_DIRS == step).all(1))[0][0])
            prev = cur
    valid[n - 1, 0] = True  # hold looks once at the current cell
    return offsets, valid, dirs


@struct.dataclass
class EnvState:
    belief: jax.Array       # [G,G] expected undetected survivors (observed prior)
    reward_map: jax.Array   # [G,G] reward copy (prior/uniform mix)
    mask: jax.Array         # [G,G] bool polygon
    looks: jax.Array        # [G,G] int look counts
    pos: jax.Array          # [N,2] int
    heading: jax.Array      # [N] int, -1 unknown
    status: jax.Array       # [N] int PENDING/ACTIVE/LEFT
    battery: jax.Array      # [N] remaining flight seconds
    freeze: jax.Array       # [N] remaining frozen decisions
    seg: jax.Array          # [N,ring,2] current segment cells (for others' view)
    join_step: jax.Array    # [N]
    leave_step: jax.Array   # [N]
    home: jax.Array         # [2] int
    pd: jax.Array
    fp: jax.Array
    total0: jax.Array       # reward map total at reset
    belief0: jax.Array      # belief total at reset
    step: jax.Array
    key: jax.Array


class SearchEnv:
    def __init__(self, cfg: EnvConfig, prior_kinds=None, real_bank=None):
        self.cfg = cfg
        self.g = cfg.grid
        self.n = cfg.n_agents
        self.r = cfg.ring
        off, val, dirs = action_tables(cfg.ring)
        self.offsets = jnp.asarray(off)
        self.valid = jnp.asarray(val)
        self.dirs = jnp.asarray(dirs)
        self.dir_vecs = jnp.asarray(_DIRS, jnp.int32)
        self.n_actions = off.shape[0]
        self.hold = self.n_actions - 1
        kinds = cfg.train_prior_kinds if prior_kinds is None else prior_kinds
        self.prior_kinds = jnp.asarray(kinds, jnp.int32)
        if maps.REAL_KIND in kinds and real_bank is None:
            raise ValueError("prior kind 4 needs real_bank")
        self.real_bank = None if real_bank is None else jnp.asarray(real_bank, jnp.float32)
        # global view geometry: pad grid up to a multiple of the pool size
        p = cfg.global_pool
        self.gpad = -(-self.g // p) * p
        self.gcoarse = self.gpad // p
        self.cpool = max(1, self.g // 32)

    # ------------------------------------------------------------------ reset
    def _spawn(self, key, mask, pos, alive):
        """Random in-mask cell farther than min_sep from all alive agents."""
        rr, cc = maps.cell_coords(self.g)
        far = jnp.ones_like(mask)
        for j in range(self.n):
            d = jnp.maximum(jnp.abs(rr - pos[j, 0]), jnp.abs(cc - pos[j, 1]))
            far = far & ((d > self.cfg.min_sep + 1) | ~alive[j])
        ok = mask & far
        ok = jnp.where(ok.any(), ok, mask)
        logits = jnp.where(ok.ravel(), 0.0, -jnp.inf)
        idx = jax.random.categorical(key, logits)
        return jnp.array([idx // self.g, idx % self.g], jnp.int32)

    def reset(self, key, extent=None):
        cfg = self.cfg
        ks = jax.random.split(key, 13)
        lo, hi = (cfg.extent_min, cfg.extent_max) if extent is None else extent
        mask = maps.sample_polygon(ks[0], self.g, lo, hi)
        kind = jax.random.choice(ks[1], self.prior_kinds)
        prior = maps.sample_prior(ks[2], mask, kind)
        if self.real_bank is not None:
            prior = jnp.where(kind == maps.REAL_KIND, maps.sample_real_prior(ks[2], mask, self.real_bank), prior)
        n_surv = jax.random.uniform(ks[3], minval=1.0, maxval=10.0)
        belief = prior * n_surv
        w = jax.random.uniform(ks[4], maxval=cfg.mix_max)
        uni = jnp.where(mask, 1.0, 0.0)
        uni = uni / uni.sum()
        reward_map = ((1 - w) * prior + w * uni) * n_surv

        home = self._spawn(ks[5], mask, jnp.zeros((self.n, 2), jnp.int32), jnp.zeros(self.n, bool))
        # agent 0 always starts; others start or join later
        late = jax.random.bernoulli(ks[6], cfg.p_join, (self.n,)).at[0].set(False)
        join_step = jnp.where(late, jax.random.randint(ks[7], (self.n,), 5, cfg.max_steps // 2), 0)
        leaves = jax.random.bernoulli(ks[8], cfg.p_leave, (self.n,))
        leave_step = jnp.where(leaves, jax.random.randint(ks[9], (self.n,), 10, cfg.max_steps), 10**6)

        pos = jnp.zeros((self.n, 2), jnp.int32)
        status = jnp.full((self.n,), PENDING, jnp.int32)
        battery = jnp.full((self.n,), cfg.battery_s, jnp.float32)
        state = EnvState(
            belief=belief, reward_map=reward_map, mask=mask,
            looks=jnp.zeros((self.g, self.g), jnp.int32),
            pos=pos, heading=jnp.full((self.n,), -1, jnp.int32), status=status,
            battery=battery, freeze=jnp.zeros(self.n, jnp.int32),
            seg=jnp.zeros((self.n, self.r, 2), jnp.int32),
            join_step=join_step, leave_step=leave_step, home=home,
            pd=jax.random.uniform(ks[10], minval=cfg.pd_min, maxval=cfg.pd_max),
            fp=jax.random.uniform(ks[11], maxval=cfg.fp_max),
            total0=reward_map.sum(), belief0=belief.sum(),
            step=jnp.zeros((), jnp.int32), key=ks[12],
        )
        state = self._joins(state)
        return state, self.observe(state)

    def _joins(self, state):
        """Activate pending agents whose join step has come."""
        key = state.key
        pos, status, battery, seg = state.pos, state.status, state.battery, state.seg
        for i in range(self.n):
            key, k = jax.random.split(key)
            go = (status[i] == PENDING) & (state.step >= state.join_step[i])
            p = self._spawn(k, state.mask, pos, status == ACTIVE)
            travel = self._dist(p, state.home) * self.cfg.t_straight
            pos = pos.at[i].set(jnp.where(go, p, pos[i]))
            seg = seg.at[i].set(jnp.where(go, jnp.broadcast_to(p, (self.r, 2)), seg[i]))
            battery = battery.at[i].set(jnp.where(go, battery[i] - travel, battery[i]))
            status = status.at[i].set(jnp.where(go, ACTIVE, status[i]))
        return state.replace(pos=pos, status=status, battery=battery, seg=seg, key=key)

    @staticmethod
    def _dist(a, b):
        d = (a - b).astype(jnp.float32)
        return jnp.sqrt((d**2).sum(-1))

    # ----------------------------------------------------------------- masks
    def paths(self, pos):
        """Absolute segment cells for every action: [N, A, ring, 2]."""
        return pos[:, None, None, :] + self.offsets[None]

    def action_mask(self, state):
        """[N, A] legal actions. Hold is always legal."""
        g = self.g
        paths = self.paths(state.pos)                       # [N,A,R,2]
        end = paths[:, :, -1]                               # [N,A,2]
        inb = ((end >= 0) & (end < g)).all(-1)
        # polygon plus 1-cell margin at the endpoint
        m = state.mask
        dil = jax.lax.reduce_window(m.astype(jnp.int32), 0, jax.lax.max, (3, 3), (1, 1), "SAME") > 0
        endc = jnp.clip(end, 0, g - 1)
        in_poly = dil[endc[..., 0], endc[..., 1]] & inb
        # separation against other active agents' current cells
        active = state.status == ACTIVE
        d = jnp.abs(paths[:, :, :, None, :] - state.pos[None, None, None, :, :]).max(-1)  # [N,A,R,N]
        other = active[None, :] & ~jnp.eye(self.n, dtype=bool)                          # [N,N]
        close = ((d <= self.cfg.min_sep) & other[:, None, None, :]).any((2, 3))
        ok = in_poly & ~close
        return ok.at[:, self.hold].set(True)

    # ------------------------------------------------------------------ step
    def step(self, state: EnvState, actions):
        cfg, g, n = self.cfg, self.g, self.n
        key, k_delay, k_noise, k_det = jax.random.split(state.key, 4)
        active = state.status == ACTIVE
        legal = self.action_mask(state)
        a = jnp.where(legal[jnp.arange(n), actions], actions, self.hold)
        delayed = jax.random.bernoulli(k_delay, cfg.p_delay, (n,))
        frozen = state.freeze > 0
        a = jnp.where(active & ~delayed & ~frozen, a, self.hold)

        # Sequential time-aligned deconfliction: agent i yields to agents < i
        all_paths = self.paths(state.pos)                   # [N,A,R,2]
        chosen = all_paths[jnp.arange(n), a]                # [N,R,2]
        hold_path = jnp.broadcast_to(state.pos[:, None, :], (n, self.r, 2))
        final = []
        for i in range(n):
            p = chosen[i]
            for j in range(i):
                dj = jnp.abs(p - final[j]).max(-1)          # [R]
                clash = (dj <= cfg.min_sep).any() & active[j] & active[i]
                a = a.at[i].set(jnp.where(clash, self.hold, a[i]))
                p = jnp.where(clash, hold_path[i], p)
            final.append(p)
        final = jnp.stack(final)                            # [N,R,2]

        # Looks: valid offsets of the executed action, active and not frozen/delayed
        lookers = active & ~frozen
        valid = self.valid[a] & lookers[:, None]            # [N,R]
        fc = jnp.clip(final, 0, g - 1)
        flat = fc[..., 0] * g + fc[..., 1]
        counts = jnp.zeros(g * g, jnp.int32).at[flat.ravel()].add(valid.ravel().astype(jnp.int32))
        counts = counts.reshape(g, g)
        miss = (1.0 - state.pd) ** counts
        gained = state.reward_map * (1.0 - miss)
        reward = gained.sum() / state.total0
        reward_map = state.reward_map * miss
        belief = state.belief * miss
        looks = state.looks + counts

        # Detection events freeze the drone for re-look and confirmation
        rm_cells = state.reward_map[fc[..., 0], fc[..., 1]]                    # [N,R]
        lam = (rm_cells * (1.0 - (1.0 - state.pd)) * valid).sum(-1)
        p_cand = jnp.clip(1.0 - jnp.exp(-lam) + state.fp * lookers, 0.0, 1.0)
        cand = jax.random.bernoulli(k_det, p_cand) & lookers
        freeze = jnp.where(cand, cfg.freeze_k, jnp.maximum(state.freeze - 1, 0))

        # Flight time: per-move straight/diagonal time plus turn penalties
        moved = a != self.hold
        mdirs = self.dirs[a]                                 # [N,R]
        diag = (mdirs % 2) == 1
        t_move = jnp.where(diag, cfg.t_diag, cfg.t_straight)
        prev = jnp.concatenate([state.heading[:, None], mdirs[:, :-1]], 1)
        turn = jnp.abs(((mdirs - prev + 4) % 8) - 4) * (prev >= 0)
        noise = 1.0 + cfg.t_noise * jax.random.normal(k_noise, (n, self.r))
        t_seg = ((t_move + cfg.t_turn * turn) * noise).sum(-1)
        t_seg = jnp.where(moved, t_seg, self.r * cfg.t_straight)
        battery = jnp.where(active, state.battery - t_seg, state.battery)
        heading = jnp.where(moved, mdirs[:, -1], state.heading)
        pos = jnp.where(active[:, None], final[:, -1], state.pos)

        # Battery rule and recall: active agents leave (RTL)
        step = state.step + 1
        ret = self._dist(pos, state.home[None]) * cfg.t_straight + cfg.reserve_s
        leave = active & ((battery <= ret) | (step >= state.leave_step))
        status = jnp.where(leave, LEFT, state.status)

        state = state.replace(
            belief=belief, reward_map=reward_map, looks=looks, pos=pos, heading=heading,
            status=status, battery=battery, freeze=freeze, seg=final, step=step, key=key,
        )
        state = self._joins(state)
        live = (state.status != LEFT).any()
        done = (step >= cfg.max_steps) | ~live | (belief.sum() < cfg.belief_stop * state.belief0)
        return state, self.observe(state), reward, done

    # ------------------------------------------------------------ observation
    def _belief_scaled(self, belief, mask):
        bmax = jnp.maximum(jnp.where(mask, belief, 0.0).max(), 1e-12)
        v = 1.0 + jnp.log(belief / bmax + 1e-6) / jnp.log(1e6)
        return jnp.where(mask, jnp.clip(v, 0.0, 1.0), 0.0)

    def observe(self, state):
        cfg, g, n = self.cfg, self.g, self.n
        active = state.status == ACTIVE
        bel = self._belief_scaled(state.belief, state.mask)
        maskf = state.mask.astype(jnp.float32)
        looksf = jnp.minimum(state.looks, 4).astype(jnp.float32) / 4.0

        # drones and their current segments, one map per agent excluding itself
        segc = jnp.clip(state.seg, 0, g - 1)
        def agent_map(i_excl):
            m = jnp.zeros((g, g))
            for j in range(n):
                w = jnp.where((j != i_excl) & active[j], 1.0, 0.0)
                m = m.at[segc[j, :, 0], segc[j, :, 1]].max(0.5 * w)
                m = m.at[state.pos[j, 0], state.pos[j, 1]].max(w)
            return m
        others = jax.vmap(agent_map)(jnp.arange(n))          # [N,G,G]

        # Local crop
        h = cfg.crop // 2
        base = jnp.stack([bel, maskf, looksf], -1)           # [G,G,3]
        base_p = jnp.pad(base, ((h, h), (h, h), (0, 0)))
        oth_p = jnp.pad(others, ((0, 0), (h, h), (h, h)))
        def crop(i):
            b = jax.lax.dynamic_slice(base_p, (state.pos[i, 0], state.pos[i, 1], 0), (cfg.crop, cfg.crop, 3))
            o = jax.lax.dynamic_slice(oth_p[i], (state.pos[i, 0], state.pos[i, 1]), (cfg.crop, cfg.crop))
            return jnp.concatenate([b, o[..., None]], -1)
        local = jax.vmap(crop)(jnp.arange(n))                # [N,c,c,4]

        # Global view: pool to coarse cells, then a 32x32 canvas centered on the drone
        p, gp, gc, gs = cfg.global_pool, self.gpad, self.gcoarse, cfg.global_size
        def pool(x, k, size):
            x = jnp.pad(x, ((0, size * k - x.shape[0]), (0, size * k - x.shape[1])))
            return x.reshape(size, k, size, k).sum((1, 3))
        bsum = pool(jnp.where(state.mask, state.belief, 0.0), p, gc)
        bglob = self._belief_scaled(bsum, bsum > 0)
        mglob = pool(maskf, p, gc) / (p * p)
        oglob = jax.vmap(lambda o: jnp.minimum(pool(o, p, gc), 1.0))(others)
        half = gs // 2
        gb = jnp.pad(jnp.stack([bglob, mglob], -1), ((half, half), (half, half), (0, 0)))
        ob = jnp.pad(oglob, ((0, 0), (half, half), (half, half)))
        def gview(i):
            c = state.pos[i] // p
            v = jax.lax.dynamic_slice(gb, (c[0], c[1], 0), (gs, gs, 2))
            o = jax.lax.dynamic_slice(ob[i], (c[0], c[1]), (gs, gs))
            return jnp.concatenate([v, o[..., None]], -1)
        glob = jax.vmap(gview)(jnp.arange(n))                # [N,32,32,3]

        # Critic world map: whole grid pooled to 32x32
        cp = self.cpool
        cs = -(-g // cp)
        wb = pool(jnp.where(state.mask, state.belief, 0.0), cp, cs)
        wbel = self._belief_scaled(wb, wb > 0)
        wmask = pool(maskf, cp, cs) / (cp * cp)
        allpos = jnp.zeros((g, g)).at[state.pos[:, 0], state.pos[:, 1]].max(active.astype(jnp.float32))
        wpos = jnp.minimum(pool(allpos, cp, cs), 1.0)
        world = jnp.stack([wbel, wmask, wpos], -1)
        world = jnp.pad(world, ((0, 32 - cs), (0, 32 - cs), (0, 0)))  # [32,32,3]

        ret = self._dist(state.pos, state.home[None]) * cfg.t_straight
        scal = jnp.stack([
            state.battery / cfg.battery_s,
            ret / cfg.battery_s,
            jnp.broadcast_to(1.0 - state.step / cfg.max_steps, (n,)),
            jnp.broadcast_to(state.belief.sum() / state.belief0, (n,)),
            jnp.broadcast_to(active.sum() / n, (n,)),
            (state.freeze > 0).astype(jnp.float32),
            jnp.broadcast_to(state.pd, (n,)),
            active.astype(jnp.float32),
            (state.home[0] - state.pos[:, 0]) / g,
            (state.home[1] - state.pos[:, 1]) / g,
            state.pos[:, 0] / g,
            state.pos[:, 1] / g,
        ], -1)                                               # [N,12]

        return {
            "local": local, "global": glob, "scalars": scal,
            "world": world, "action_mask": self.action_mask(state), "active": active,
        }


N_SCALARS = 12
