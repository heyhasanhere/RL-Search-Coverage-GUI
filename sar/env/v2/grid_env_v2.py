"""Version 2 multi-drone search grid environment (plan sections 7, 9.1).

Built beside `sar/env/grid_env.py`, which stays untouched so the version 1
checkpoints in `runs/main_r4` keep loading and evaluating.

Changes against version 1 (plan section 7.1):
  - grid up to 192 cells; curriculum 32 -> 64 -> 128, evaluation to 192
  - two altitude levels: low (1 cell footprint) and high (3 x 3 cells)
  - 66 actions: (32 ring-4 end cells or hold) x 2 levels
  - per-cell Pd from Pd[level, land cover class, canopy class]
  - separation per plan section 7.2: same level keeps min_sep cells; different
    levels may cross only where their heights differ by at least 20 m
  - cells whose altitude floor is above the high level are masked at both levels
  - battery sorties: return, 180 s battery swap on the ground, relaunch
  - SIH-calibrated turn time (1.02 s per 45 degrees), climb/descend on a level change
  - observation additions per plan section 7.3: Pd channels per level, drone
    levels, sortie state
  - global view 32 x 32 at 12 cells per pixel (pooled coarser on smaller grids)

Unchanged from version 1: the Poisson belief and its update, the reward
(expected survivors detected over the prior total), detection events that freeze
a drone for the re-look, the reward-map mix, fleet changes, one-step delays.
"""


import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from sar.config_v2 import HIGH, LOW, N_LEVELS, EnvConfigV2
from sar.env import grid_env as v1
from sar.env.v2 import maps_v2
from sar.config_v2 import load_pd_table
from sar.env.v2.pd_table import PdTable

# Agent status values, same codes as version 1
PENDING, ACTIVE, LEFT = v1.PENDING, v1.ACTIVE, v1.LEFT

# Ground states for the sortie machine
AIRBORNE, SWAPPING, DONE = 0, 1, 2

_DIRS = v1._DIRS


def action_tables(ring: int):
    """Ring offsets per level: 2 x (A0, ring, 2) cumulative cell offsets.

    `off[lv, a, k]` is the k-th cell of segment base action `a` flown at level
    `lv`. `valid[lv, a, k]` marks which offsets produce looks; level 0 looks at
    the offset cell itself, level 1 widens it to a 3 x 3 block. `foot[lv, a, k]`
    is the list of looked-at cells (1 or 9) per offset. The last base action of
    each level is hold.
    """
    off, val, dirs = v1.action_tables(ring)
    a0 = off.shape[0]
    nb = np.array([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], np.int32)
    # Both levels carry the same 3 x 3 block; the low level activates only the
    # centre slot (index 4), so its footprint is exactly one cell (plan 7.2).
    foot_hi = (off[:, :, None, :] + nb[None, None, :, :]).astype(np.int32)  # [A0,R,9,2]
    foot = np.stack([foot_hi, foot_hi], 0)           # [2,A0,R,9,2]
    # slot mask per level: low = centre only, high = all nine.
    slot = np.zeros((2, 1, 1, 9), np.float32)
    slot[HIGH, 0, 0, :] = 1.0
    slot[LOW, 0, 0, 4] = 1.0
    return (np.stack([off, off], 0), np.stack([val, val], 0),
            np.stack([dirs, dirs], 0), foot, slot)


@struct.dataclass
class EnvStateV2:
    belief: jax.Array        # [G,G] expected undetected survivors (observed prior)
    reward_map: jax.Array    # [G,G] reward copy (prior/uniform mix)
    mask: jax.Array          # [G,G] bool searchable polygon
    looks: jax.Array         # [G,G] int look counts
    pos: jax.Array           # [N,2] int
    heading: jax.Array       # [N] int, -1 unknown
    status: jax.Array        # [N] int PENDING/ACTIVE/LEFT
    level: jax.Array         # [N] int, current altitude level of each drone
    battery: jax.Array       # [N] remaining flight seconds this sortie
    sorties: jax.Array       # [N] completed sorties
    ground: jax.Array        # [N] int AIRBORNE/SWAPPING/DONE
    swap_left: jax.Array     # [N] seconds left of the battery swap
    freeze: jax.Array        # [N] remaining frozen decisions
    seg: jax.Array           # [N,ring,2] current segment cells (for others' view)
    join_step: jax.Array     # [N]
    leave_step: jax.Array    # [N]
    home: jax.Array          # [2] int
    pd: jax.Array            # [2,G,G] Pd per level (table x per-episode jitter)
    fp: jax.Array
    floor_m: jax.Array       # [G,G] altitude floor per cell (dem + obstacle + clearance)
    dem_m: jax.Array         # [G,G] ground elevation per cell
    total0: jax.Array        # reward map total at reset
    belief0: jax.Array       # belief total at reset
    step: jax.Array
    key: jax.Array


class SearchEnvV2:
    """JAX, jit/vmap friendly. One instance = one episode family."""

    def __init__(self, cfg: EnvConfigV2, pd_table: PdTable | None = None, prior_bank=None):
        self.cfg = cfg
        self.g = cfg.grid
        self.n = cfg.n_agents
        self.r = cfg.ring
        off, val, dirs, foot, slot = action_tables(cfg.ring)
        self.offsets = jnp.asarray(off)          # [2,A0,R,2]
        self.valid = jnp.asarray(val)            # [2,A0,R]
        self.dirs = jnp.asarray(dirs)            # [2,A0,R]
        self.foot = jnp.asarray(foot)            # [2,A0,R,9,2]
        self.slot = jnp.asarray(slot)            # [2,1,1,9] active footprint slots
        # flat per-action view [n_actions,9]: action = level*a0 + base. Policies
        # index this directly, so they need no gather over the level axis.
        slot_flat = np.repeat(slot.reshape(2, 1, 9)[:, :, :], off.shape[1], axis=1)
        self.slot_table = jnp.asarray(slot_flat.reshape(2 * off.shape[1], 9))
        self.dir_vecs = jnp.asarray(_DIRS, jnp.int32)
        self.a0 = off.shape[1]                   # 33 base actions
        self.n_actions = N_LEVELS * self.a0      # 66
        self.hold = cfg.hold_action              # hold at the low level
        self.pd_table = pd_table if pd_table is not None else (cfg.pd_table or load_pd_table())
        self.prior_bank = prior_bank
        self.prior_kinds = jnp.asarray(cfg.train_prior_kinds, jnp.int32)
        self._build_terrain_bank()
        # Global view geometry (plan section 7.1): 32 x 32 pixels at 12 cells per pixel,
        # so the view spans 384 cells. The coarse grid is gpad/12; it is centred in
        # the 32 x 32 canvas (padded for grids smaller than 384, cropped above).
        self.global_pool = cfg.global_pool
        self.gpad = -(-self.g // self.global_pool) * self.global_pool
        self.gcoarse = self.gpad // self.global_pool
        self.cpool = max(1, -(-self.g // 32))
        # climb/descend time between the two levels (plan sections 7.1, 9.1)
        self.t_level = abs(cfg.level_m[HIGH] - cfg.level_m[LOW]) / cfg.climb_rate

    # ------------------------------------------------------------------ reset
    def _spawn(self, key, mask, pos, alive):
        """Random in-mask cell farther than min_sep from all alive agents."""
        g = self.g
        rr, cc = maps_v2.cell_coords(g)
        far = jnp.ones_like(mask)
        for j in range(self.n):
            d = jnp.maximum(jnp.abs(rr - pos[j, 0]), jnp.abs(cc - pos[j, 1]))
            far = far & ((d > self.cfg.min_sep + 1) | ~alive[j])
        ok = mask & far
        ok = jnp.where(ok.any(), ok, mask)
        logits = jnp.where(ok.ravel(), 0.0, -jnp.inf)
        idx = jax.random.categorical(key, logits)
        return jnp.array([idx // g, idx % g], jnp.int32)

    def _build_terrain_bank(self):
        """Precompute the per-cell terrain layers once, on the host.

        The real prior bank replaces the synthetic one when it is available, so
        the other worker's region data drops in without touching the env. Index
        happens with a traced seed, which keeps `reset` vmappable.
        """
        if self.prior_bank is not None:
            self.terrain = {k: self._as_bank(jnp.asarray(v))
                            for k, v in self.prior_bank.items()
                            if k in ("landcover", "canopy", "dem_m", "floor_m")}
        else:
            n_maps = 64
            cols = [maps_v2.synthetic_terrain_np(0, self.g, seed=i)
                    for i in range(n_maps)]
            self.terrain = {
                k: jnp.asarray(np.stack([c[k] for c in cols], 0))
                for k in ("landcover", "canopy", "dem_m", "floor_m")
            }

    def _as_bank(self, arr):
        """One map as [1, G, G] so a single-region bank indexes like the bank.

        Tier R builds a one-case bank of bare [G, G] layers; the synthetic bank
        is [N, G, G]. Adding the leading map axis here means `_terrain` works
        the same for both.
        """
        return arr if arr.ndim == 3 else arr[None]

    def _terrain(self, seed: jax.Array):
        """Per-cell layers, indexed by a traced seed so reset stays vmappable.

        The layers are precomputed once on the host in __init__ (a small fixed
        bank) and indexed here, so no Python int() is needed at trace time. The
        real prior bank, when present, replaces the synthetic bank.
        """
        idx = jnp.asarray(seed) % self.terrain["landcover"].shape[0]
        return (self.terrain["landcover"][idx], self.terrain["canopy"][idx],
                self.terrain["dem_m"][idx], self.terrain["floor_m"][idx])

    def _prior(self, key, mask, kind, seed):
        """Prior intensity map, normalized inside the mask.

        The bank check is a Python-level test (the bank is a static attribute),
        so this stays vmappable; only the bank index is traced.
        """
        if self.prior_bank is not None and "prior" in self.prior_bank:
            # Same region index as `_terrain`, so the prior and the terrain
            # layers always describe the same place.
            bank_prior = self._as_bank(jnp.asarray(self.prior_bank["prior"]))
            idx = jnp.asarray(seed) % bank_prior.shape[0]
            p = bank_prior[idx]
            p = jnp.where(mask, p, 0.0)
            return p / jnp.maximum(p.sum(), 1e-12)
        return maps_v2.sample_prior_v2(key, mask, kind)

    def reset(self, key, extent=None, seed: int | None = None):
        cfg = self.cfg
        ks = jax.random.split(key, 13)
        lo, hi = (cfg.extent_min, cfg.extent_max) if extent is None else extent
        mask = maps_v2.sample_polygon(ks[0], self.g, lo, hi)
        kind = jax.random.choice(ks[1], self.prior_kinds)
        # The terrain index is a traced value: the bank is precomputed on the
        # host, so reset is vmappable over a batch of keys. A PRNG key is two
        # uint32 words, so fold it to a single scalar index first.
        tseed = ks[12].astype(jnp.uint32).sum() if seed is None \
            else jnp.asarray(seed, jnp.uint32)
        prior = self._prior(ks[2], mask, kind, tseed)
        n_surv = jax.random.uniform(ks[3], minval=1.0, maxval=10.0)
        belief = prior * n_surv
        w = jax.random.uniform(ks[4], maxval=cfg.mix_max)
        uni = jnp.where(mask, 1.0, 0.0)
        uni = uni / uni.sum()
        reward_map = ((1 - w) * prior + w * uni) * n_surv

        # Terrain layers, per-cell Pd maps and the unsearchable mask (plan 7.1, 9.1).
        lc, cp, dem, floor = self._terrain(tseed)
        pd_maps = jnp.asarray(self.pd_table.level_maps(lc, cp, cfg.pd_scale))
        jitter = jax.random.uniform(ks[10], minval=cfg.pd_min_scale, maxval=cfg.pd_max_scale)
        pd = jnp.clip(pd_maps * jitter, 0.0, 1.0)
        # floor above the high level -> unsearchable from the air (plan section 9.1)
        blocked = floor > (cfg.level_m[HIGH] + cfg.clearance_m)
        mask = mask & ~blocked

        home = self._spawn(ks[5], mask, jnp.zeros((self.n, 2), jnp.int32), jnp.zeros(self.n, bool))
        late = jax.random.bernoulli(ks[6], cfg.p_join, (self.n,)).at[0].set(False)
        join_step = jnp.where(late, jax.random.randint(ks[7], (self.n,), 5, cfg.max_steps // 2), 0)
        leaves = jax.random.bernoulli(ks[8], cfg.p_leave, (self.n,))
        leave_step = jnp.where(leaves, jax.random.randint(ks[9], (self.n,), 10, cfg.max_steps), 10**6)

        state = EnvStateV2(
            belief=belief, reward_map=reward_map, mask=mask,
            looks=jnp.zeros((self.g, self.g), jnp.int32),
            pos=jnp.zeros((self.n, 2), jnp.int32), heading=jnp.full((self.n,), -1, jnp.int32),
            status=jnp.full((self.n,), PENDING, jnp.int32),
            level=jnp.zeros((self.n,), jnp.int32),
            battery=jnp.full((self.n,), cfg.battery_s, jnp.float32),
            sorties=jnp.zeros((self.n,), jnp.int32),
            ground=jnp.full((self.n,), AIRBORNE, jnp.int32),
            swap_left=jnp.zeros((self.n,), jnp.float32),
            freeze=jnp.zeros(self.n, jnp.int32),
            seg=jnp.zeros((self.n, self.r, 2), jnp.int32),
            join_step=join_step, leave_step=leave_step, home=home,
            pd=pd, fp=jax.random.uniform(ks[11], maxval=cfg.fp_max),
            floor_m=floor, dem_m=dem,
            total0=reward_map.sum(), belief0=belief.sum(),
            step=jnp.zeros((), jnp.int32), key=ks[12],
        )
        state = self._joins(state)
        return state, self.observe(state)

    def _joins(self, state):
        """Activate pending agents whose join step has come (at the low level)."""
        key = state.key
        pos, status, battery, seg = state.pos, state.status, state.battery, state.seg
        level, ground = state.level, state.ground
        for i in range(self.n):
            key, k = jax.random.split(key)
            go = (status[i] == PENDING) & (state.step >= state.join_step[i])
            p = self._spawn(k, state.mask, pos, status == ACTIVE)
            travel = self._dist(p, state.home) * self.cfg.t_straight
            pos = pos.at[i].set(jnp.where(go, p, pos[i]))
            seg = seg.at[i].set(jnp.where(go, jnp.broadcast_to(p, (self.r, 2)), seg[i]))
            battery = battery.at[i].set(jnp.where(go, battery[i] - travel, battery[i]))
            status = status.at[i].set(jnp.where(go, ACTIVE, status[i]))
            level = level.at[i].set(jnp.where(go, LOW, level[i]))
            ground = ground.at[i].set(jnp.where(go, AIRBORNE, ground[i]))
        return state.replace(pos=pos, status=status, battery=battery, seg=seg,
                             level=level, ground=ground, key=key)

    @staticmethod
    def _dist(a, b):
        d = (a - b).astype(jnp.float32)
        return jnp.sqrt((d**2).sum(-1))

    # ----------------------------------------------------------------- heights
    def level_heights(self, state):
        """Absolute height above ground of each drone: floor + level height."""
        pos = jnp.clip(state.pos, 0, self.g - 1)
        return state.floor_m[pos[:, 0], pos[:, 1]] + jnp.asarray(self.cfg.level_m)[state.level]

    def _height_gap_ok(self, state):
        """[N,N] can two drones cross here? Same level never; different levels
        only when their heights differ by at least 20 m (plan section 7.2)."""
        gap = jnp.abs(self.level_heights(state)[:, None] - self.level_heights(state)[None, :])
        return gap >= self.cfg.min_level_gap_m

    # ----------------------------------------------------------------- masks
    def paths(self, pos):
        """Absolute segment cells for all level/base actions: [N,2,A0,R,2]."""
        return pos[:, None, None, None, :] + self.offsets[None]

    def action_mask(self, state):
        """[N, 66] legal actions. Hold at the current level is always legal."""
        cfg, g = self.cfg, self.g
        lev_a = jnp.arange(cfg.n_actions) // self.a0              # [66]
        base_a = jnp.arange(cfg.n_actions) % self.a0              # [66]
        ends = state.pos[:, None, :] + self.offsets[lev_a, base_a, -1]   # [N,66,2]
        inb = ((ends >= 0) & (ends < g)).all(-1)
        dil = jax.lax.reduce_window(state.mask.astype(jnp.int32), 0, jax.lax.max,
                                    (3, 3), (1, 1), "SAME") > 0
        endc = jnp.clip(ends, 0, g - 1)
        in_poly = dil[endc[..., 0], endc[..., 1]] & inb

        # Separation (plan section 7.2) against the other drones' current cells.
        paths_a = self.paths(state.pos)[:, lev_a, base_a]          # [N,66,R,2]
        d = jnp.abs(paths_a[:, :, :, None, :] - state.pos[None, None, None, :, :]).max(-1)
        active = state.status == ACTIVE
        other = active[None, :] & ~jnp.eye(self.n, dtype=bool)    # [N,N]
        gap_ok = self._height_gap_ok(state)                       # [N,N]
        same_level = state.level[:, None] == state.level[None, :]
        close = (d <= cfg.min_sep) & other[:, None, None, :] & (same_level | ~gap_ok)[:, None, None, :]
        ok = in_poly & ~close.any((2, 3))
        return ok.at[jnp.arange(self.n), self.hold + state.level * self.a0].set(True)

    # ------------------------------------------------------------------ step
    def step(self, state: EnvStateV2, actions):
        """One decision for every drone. Returns (state, obs, reward, done)."""
        cfg, g, n = self.cfg, self.g, self.n
        key, k_delay, k_noise, k_det = jax.random.split(state.key, 4)
        active = state.status == ACTIVE
        airborne = active & (state.ground == AIRBORNE)
        legal = self.action_mask(state)
        a = jnp.where(legal[jnp.arange(n), actions], actions, self.hold)
        delayed = jax.random.bernoulli(k_delay, cfg.p_delay, (n,))
        frozen = state.freeze > 0
        # delayed, frozen, or still-swapping drones hold at their current level
        hold_a = self.hold + state.level * self.a0
        move = airborne & ~delayed & ~frozen
        lev_a = jnp.where(move, a // self.a0, state.level)
        base_a = jnp.where(move, a % self.a0, self.a0 - 1)

        # Sequential time-aligned deconfliction: agent i yields to agents < i.
        all_paths = self.paths(state.pos)                         # [N,2,A0,R,2]
        hold_path = jnp.broadcast_to(state.pos[:, None, :], (n, self.r, 2))
        final, heights = [], self.level_heights(state)
        for i in range(n):
            p = all_paths[i, lev_a[i], base_a[i]]
            for j in range(i):
                dj = jnp.abs(p - final[j]).max(-1)
                gap = jnp.abs(heights[i] - heights[j])
                clash = (dj <= cfg.min_sep).any() & airborne[j] & airborne[i] & (
                    (state.level[i] == state.level[j]) | (gap < cfg.min_level_gap_m))
                base_a = base_a.at[i].set(jnp.where(clash, self.a0 - 1, base_a[i]))
                lev_a = lev_a.at[i].set(jnp.where(clash, state.level[i], lev_a[i]))
                p = jnp.where(clash, hold_path[i], p)
            final.append(p)
        final = jnp.stack(final)                                  # [N,R,2]
        base_a = jnp.where(move, base_a, self.a0 - 1)             # a clashing drone holds

        # ---- looks: scatter (cell, Pd) pairs of every looked cell into flat grids
        lev_one = lev_a[:, None, None]                            # [N,1,1]
        foot = self.foot[lev_a, base_a] + state.pos[:, None, None, :]   # [N,R,9,2] absolute
        vv = (self.valid[lev_a, base_a] & move[:, None])[:, :, None]   # [N,R,1]
        slot_n = self.slot[lev_a, 0, 0]                           # [N,9]
        inb = ((foot >= 0) & (foot < g)).all(-1)                  # [N,R,9]
        w = (vv.astype(jnp.float32) * slot_n[:, None, :]) * inb   # [N,R,9]
        fc = jnp.clip(foot, 0, g - 1)
        pd_cells = self._pd_gather(state.pd, lev_one, fc)              # [N,R,9]
        flat = fc[..., 0] * g + fc[..., 1]                       # [N,R,9]
        wf = w
        # A cell looked at several times in one step (or from several slots) is
        # looked at once per look: multiply the (1 - Pd) miss factors.
        miss_n = jnp.ones(g * g, jnp.float32)
        miss_n = miss_n.at[flat.ravel()].mul(
            jnp.where(wf.ravel() > 0, 1.0 - pd_cells.ravel(), 1.0))
        counts_n = jnp.zeros(g * g, jnp.int32).at[flat.ravel()].add(
            (wf.ravel() > 0).astype(jnp.int32))
        miss = miss_n.reshape(g, g)
        counts = counts_n.reshape(g, g)

        gained = state.reward_map * (1.0 - miss)
        reward = gained.sum() / state.total0
        reward_map = state.reward_map * miss
        belief = state.belief * miss
        looks = state.looks + counts

        # Detection events freeze the drone for the re-look and confirmation.
        pc = jnp.clip(final, 0, g - 1)
        pd_centre = self._pd_gather(state.pd, lev_a[:, None], pc)     # [N,R]
        valid_moves = self.valid[lev_a, base_a] & move[:, None]     # [N,R]
        rm_cells = state.reward_map[pc[..., 0], pc[..., 1]]        # [N,R]
        lam = (rm_cells * pd_centre * valid_moves).sum(-1)
        p_cand = jnp.clip(1.0 - jnp.exp(-lam) + state.fp * airborne, 0.0, 1.0)
        cand = jax.random.bernoulli(k_det, p_cand) & airborne
        freeze = jnp.where(cand, cfg.freeze_k, jnp.maximum(state.freeze - 1, 0))

        # Flight time: per-move straight/diagonal + SIH turn + level change.
        moved = move & (base_a != self.a0 - 1)
        mdirs = self.dirs[lev_a, base_a]                           # [N,R]
        t_move = jnp.where((mdirs % 2) == 1, cfg.t_diag, cfg.t_straight)
        prev = jnp.concatenate([state.heading[:, None], mdirs[:, :-1]], 1)
        turn = jnp.abs(((mdirs - prev + 4) % 8) - 4) * (prev >= 0)
        noise = 1.0 + cfg.t_noise * jax.random.normal(k_noise, (n, self.r))
        t_seg = ((t_move + cfg.t_turn * turn) * noise).sum(-1)
        t_seg = jnp.where(moved, t_seg, self.r * cfg.t_straight)
        # a level change costs the climb even when the drone also holds position
        t_seg = t_seg + jnp.where(airborne & (lev_a != state.level), self.t_level, 0.0)
        battery = jnp.where(airborne, state.battery - t_seg, state.battery)
        heading = jnp.where(moved, mdirs[:, -1], state.heading)
        pos = jnp.where(airborne[:, None], final[:, -1], state.pos)
        level = jnp.where(airborne, lev_a, state.level)

        # Battery rule, recall and the sortie machine (plan section 7.1).
        step = state.step + 1
        ret = self._dist(pos, state.home[None]) * cfg.t_straight + cfg.reserve_s
        recalled = airborne & (step >= state.leave_step)
        out_of_sorties = airborne & (state.sorties >= cfg.max_sorties)
        low = airborne & (battery <= ret)
        # A low battery starts a battery swap: the drone stays ACTIVE but is on
        # the ground until _sorties relaunches it. Recall or the last sortie ends
        # the drone for good. Either way it stops searching this step.
        swap = low & ~recalled & ~out_of_sorties
        done_out = recalled | out_of_sorties
        # only a finished drone leaves; a swapping one stays in the fleet
        status = jnp.where(done_out | (low & ~swap), LEFT, state.status)
        ground = jnp.where(swap, SWAPPING, jnp.where(done_out, DONE, state.ground))
        swap_left = jnp.where(swap, cfg.swap_s, state.swap_left)
        seg = jnp.where(airborne[:, None, None], final, state.seg)

        state = state.replace(
            belief=belief, reward_map=reward_map, looks=looks, pos=pos, heading=heading,
            status=status, level=level, battery=battery, sorties=state.sorties,
            ground=ground, swap_left=swap_left, freeze=freeze, seg=seg, step=step, key=key,
        )
        state = self._sorties(state)
        live = (state.status != LEFT).any()
        done = (step >= cfg.max_steps) | ~live | (belief.sum() < cfg.belief_stop * state.belief0)
        return state, self.observe(state), reward, done

    def _sorties(self, state):
        """Count the ground swap down and relaunch. No flight time on the ground.

        `swap_left` is mission seconds left before the new battery is in. Each env
        decision step stands for one decision, whose wall-clock cost is the segment
        time the mission clock advances by, so the swap is counted down in the same
        seconds the orchestrator would spend (plan section 7.1).
        """
        swapping = state.ground == SWAPPING
        # one decision advances the mission clock by a median segment (two levels)
        tick = self.cfg.t_straight * 2.0 + self.t_level
        swap_left = jnp.where(swapping, jnp.maximum(state.swap_left - tick, 0.0), state.swap_left)
        relaunch = swapping & (swap_left <= 0.0)
        home = jnp.broadcast_to(state.home[None, :], (self.n, 2))
        return state.replace(
            battery=jnp.where(relaunch, self.cfg.battery_s, state.battery),
            status=jnp.where(relaunch, ACTIVE, state.status),
            ground=jnp.where(relaunch, AIRBORNE, state.ground),
            level=jnp.where(relaunch, LOW, state.level),      # relaunch low
            sorties=jnp.where(relaunch, state.sorties + 1, state.sorties),
            swap_left=swap_left,
            pos=jnp.where(relaunch[:, None], home, state.pos),
            seg=jnp.where(relaunch[:, None, None], home[:, None, :], state.seg),
        )

    # ------------------------------------------------------------ observation
    def _pd_level(self, pd, level: int):
        """One level's Pd map, indexed from the end of the level axis.

        `pd` is [2, G, G] unbatched and [B, 2, G, G] when the state is vmapped,
        so the level axis has to be addressed from the end for both to work.
        A single-map bank (Tier R) can arrive as a bare [G, G] with the level
        axis already indexed away, so promote it back before taking.
        """
        pd = jnp.asarray(pd)
        if pd.ndim == 2:
            pd = pd[None, :, :]
        return jnp.take(pd, jnp.asarray(level), axis=-3)

    def _pd_gather(self, pd, level, cells):
        """Pd at `cells` for `level`: pd[..., level, r, c].

        `level` and `cells` broadcast against each other in the leading axes,
        as the unbatched advanced indexing used to. The two grid axes are
        flattened into one so the gather is a single take, which keeps the
        shape exact whether or not `pd` carries a leading batch axis.
        """
        g = self.g
        lev = jnp.asarray(level)
        shape = cells.shape[:-1]
        while lev.ndim < len(shape):
            lev = lev[..., None]
        lev = lev[..., :len(shape)]
        levb = jnp.broadcast_to(lev, shape)
        rb = jnp.clip(jnp.broadcast_to(cells[..., 0], shape), 0, g - 1)
        cb = jnp.clip(jnp.broadcast_to(cells[..., 1], shape), 0, g - 1)
        flat = (rb * g + cb).astype(jnp.int32)                 # cell -> grid index
        out = 0.0
        for lv in range(N_LEVELS):
            lm = self._pd_level(pd, lv)                        # [G,G] or [B,G,G]
            # flatten the two grid axes to one, so the gather index only has to
            # line up with a possible leading batch axis
            lmf = lm.reshape(lm.shape[:-2] + (g * g,)) if lm.ndim >= 2 else lm
            idx = flat
            while idx.ndim < lmf.ndim:
                idx = jnp.expand_dims(idx, 0)
            got = jnp.take(lmf, idx, axis=-1)                  # [..., shape]
            got = jnp.reshape(got, shape)
            out = out + jnp.where(levb == lv, got, 0.0)
        return out

    def _belief_scaled(self, belief, mask):
        bmax = jnp.maximum(jnp.where(mask, belief, 0.0).max(), 1e-12)
        v = 1.0 + jnp.log(belief / bmax + 1e-6) / jnp.log(1e6)
        return jnp.where(mask, jnp.clip(v, 0.0, 1.0), 0.0)

    def observe(self, state):
        """Local crop, global view, scalars and critic world map (plan section 7.3)."""
        cfg, g, n = self.cfg, self.g, self.n
        active = state.status == ACTIVE
        bel = self._belief_scaled(state.belief, state.mask)
        maskf = state.mask.astype(jnp.float32)
        looksf = jnp.minimum(state.looks, 4).astype(jnp.float32) / 4.0
        pd_low = jnp.clip(self._pd_level(state.pd, LOW), 0.0, 1.0)
        pd_high = jnp.clip(self._pd_level(state.pd, HIGH), 0.0, 1.0)
        blockedf = (state.floor_m > (cfg.level_m[HIGH] + cfg.clearance_m)).astype(jnp.float32)

        segc = jnp.clip(state.seg, 0, g - 1)

        def agent_map(i_excl):
            m = jnp.zeros((g, g))
            for j in range(n):
                w = jnp.where((j != i_excl) & active[j], 1.0, 0.0)
                m = m.at[segc[j, :, 0], segc[j, :, 1]].max(0.5 * w)
                m = m.at[state.pos[j, 0], state.pos[j, 1]].max(w)
            return m

        others = jax.vmap(agent_map)(jnp.arange(n))               # [N,G,G]

        # Local crop: belief, mask, looks, Pd low, Pd high, blocked, others.
        h = cfg.crop // 2
        base = jnp.stack([bel, maskf, looksf, pd_low, pd_high, blockedf], -1)
        base_p = jnp.pad(base, ((h, h), (h, h), (0, 0)))
        oth_p = jnp.pad(others, ((0, 0), (h, h), (h, h)))

        def crop(i):
            b = jax.lax.dynamic_slice(base_p, (state.pos[i, 0], state.pos[i, 1], 0),
                                      (cfg.crop, cfg.crop, 6))
            o = jax.lax.dynamic_slice(oth_p[i], (state.pos[i, 0], state.pos[i, 1]),
                                      (cfg.crop, cfg.crop))
            return jnp.concatenate([b, o[..., None]], -1)

        local = jax.vmap(crop)(jnp.arange(n))                     # [N,c,c,7]

        # Global view 32 x 32 at 12 cells per pixel (pool coarser on small grids).
        p, gc, gs = self.global_pool, self.gcoarse, cfg.global_size

        def pool(x, k, size):
            x = jnp.pad(x, ((0, size * k - x.shape[0]), (0, size * k - x.shape[1])))
            return x.reshape(size, k, size, k).sum((1, 3))

        bsum = pool(jnp.where(state.mask, state.belief, 0.0), p, gc)
        bglob = self._belief_scaled(bsum, bsum > 0)
        mglob = pool(maskf, p, gc) / (p * p)
        pdl_g = pool(pd_low, p, gc) / (p * p)
        pdh_g = pool(pd_high, p, gc) / (p * p)
        oglob = jax.vmap(lambda o: jnp.minimum(pool(o, p, gc), 1.0))(others)
        # The coarse grid is at most 32 wide; centre it in the gs x gs canvas and
        # take the window around each drone (plan section 7.1).
        org = (gs - self.gcoarse) // 2
        pad2 = ((0, 0),) * 0 + ((org, gs - self.gcoarse - org),) * 2
        gb = jnp.pad(jnp.stack([bglob, mglob, pdl_g, pdh_g], -1), pad2 + ((0, 0),))
        ob = jnp.pad(oglob, ((0, 0),) + pad2)

        def gview(i):
            c = state.pos[i] // p + org - gs // 2
            c0 = jnp.clip(c[0], 0, gb.shape[0] - gs)
            c1 = jnp.clip(c[1], 0, gb.shape[1] - gs)
            v = jax.lax.dynamic_slice(gb, (c0, c1, 0), (gs, gs, 4))
            o = jax.lax.dynamic_slice(ob[i], (c0, c1), (gs, gs))
            return jnp.concatenate([v, o[..., None]], -1)

        glob = jax.vmap(gview)(jnp.arange(n))                     # [N,32,32,5]

        # Critic world map: whole grid pooled to 32 x 32.
        cp = self.cpool
        cs = -(-g // cp)
        wb = pool(jnp.where(state.mask, state.belief, 0.0), cp, cs)
        wbel = self._belief_scaled(wb, wb > 0)
        wmask = pool(maskf, cp, cs) / (cp * cp)
        allpos = jnp.zeros((g, g)).at[state.pos[:, 0], state.pos[:, 1]].max(active.astype(jnp.float32))
        wpos = jnp.minimum(pool(allpos, cp, cs), 1.0)
        world = jnp.stack([wbel, wmask, wpos], -1)
        world = jnp.pad(world, ((0, max(0, 32 - cs)), (0, max(0, 32 - cs)), (0, 0)))[:32, :32]

        # Scalars (plan section 7.3): battery, return time, mission time, belief,
        # fleet size, freeze, each drone's level, swap state and time to relaunch.
        ret = self._dist(state.pos, state.home[None]) * cfg.t_straight
        sorting = state.ground == SWAPPING
        relaunch_frac = jnp.where(sorting, 1.0 - state.swap_left / cfg.swap_s, 0.0)
        scal = jnp.stack([
            state.battery / cfg.battery_s,
            ret / cfg.battery_s,
            jnp.broadcast_to(1.0 - state.step / cfg.max_steps, (n,)),
            jnp.broadcast_to(state.belief.sum() / state.belief0, (n,)),
            jnp.broadcast_to(active.sum() / n, (n,)),
            (state.freeze > 0).astype(jnp.float32),
            state.level.astype(jnp.float32),
            sorting.astype(jnp.float32),
            relaunch_frac,
            state.sorties.astype(jnp.float32) / max(cfg.max_sorties, 1),
            active.astype(jnp.float32),
            (state.home[0] - state.pos[:, 0]) / g,
            (state.home[1] - state.pos[:, 1]) / g,
            state.pos[:, 0] / g,
            state.pos[:, 1] / g,
            (1.0 - state.dem_m[jnp.clip(state.pos[:, 0], 0, g - 1),
                               jnp.clip(state.pos[:, 1], 0, g - 1)] / 120.0),
        ], -1)                                                    # [N,16]

        return {"local": local, "global": glob, "scalars": scal, "world": world,
                "action_mask": self.action_mask(state), "active": active}

    # ---------------------------------------------------------------- helpers
    def zero_state(self, **overrides):
        """Build a zero EnvStateV2 (used by tests and the search planner)."""
        g, n = self.g, self.n
        z = jnp.zeros((g, g), jnp.float32)
        defaults = dict(
            belief=z, reward_map=z, mask=jnp.ones((g, g), bool),
            looks=jnp.zeros((g, g), jnp.int32),
            pos=jnp.zeros((n, 2), jnp.int32), heading=jnp.full((n,), -1, jnp.int32),
            status=jnp.full((n,), PENDING, jnp.int32), level=jnp.zeros((n,), jnp.int32),
            battery=jnp.full((n,), self.cfg.battery_s, jnp.float32),
            sorties=jnp.zeros((n,), jnp.int32), ground=jnp.full((n,), AIRBORNE, jnp.int32),
            swap_left=jnp.zeros((n,), jnp.float32), freeze=jnp.zeros((n,), jnp.int32),
            seg=jnp.zeros((n, self.r, 2), jnp.int32), join_step=jnp.zeros((n,), jnp.int32),
            leave_step=jnp.full((n,), 10**6, jnp.int32), home=jnp.zeros((2,), jnp.int32),
            pd=jnp.ones((N_LEVELS, g, g), jnp.float32), fp=jnp.float32(0.0),
            floor_m=jnp.zeros((g, g), jnp.float32), dem_m=jnp.zeros((g, g), jnp.float32),
            total0=jnp.float32(1.0), belief0=jnp.float32(1.0),
            step=jnp.zeros((), jnp.int32), key=jax.random.PRNGKey(0),
        )
        defaults.update(overrides)
        return EnvStateV2(**defaults)


N_SCALARS_V2 = 16