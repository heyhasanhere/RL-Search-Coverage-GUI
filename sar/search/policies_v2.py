"""Lawnmower, greedy and look-ahead planner for env v2 (plan section 8.1).

Same three baselines as version 1, extended to both altitude levels: lawnmower
sweeps at the low level, greedy takes the best immediate expected detections over
all 66 actions, and the look-ahead planner looks two segments ahead over both
levels. Each policy maps (env, state, key) -> actions [N] for one env; batch with
vmap.
"""

import jax
import jax.numpy as jnp

from sar.config_v2 import LOW
from sar.env.v2.grid_env_v2 import ACTIVE, SearchEnvV2

NEG = -1e9
BLOCK = 8          # block size for the long-range pull potential


def footprint_cells(env: SearchEnvV2, pos, actions):
    """Absolute looked-at cells of one action per agent.

    Returns foot [N,A,R,9,2], the bool weight [N,A,R,9], the slot mask [N,A,9]
    and the level [N,A].
    """
    g = env.g
    lev = actions // env.a0
    base = actions % env.a0
    foot = env.foot[lev, base] + pos[:, None, None, None, :]       # [N,A,R,9,2]
    slot = env.slot_table[: env.n_actions, :]                      # [A,9] static per level/action
    valid = env.valid[lev, base]                                    # [N,A,R]
    inb = ((foot >= 0) & (foot < g)).all(-1)                        # [N,A,R,9]
    w = valid[:, :, :, None] & inb
    return foot, w, slot[None, :, :], lev                           # [N,A,9] slot


def seg_value(env: SearchEnvV2, belief, pd, pos, actions):
    """Expected detections of each action: [N, 66]."""
    foot, w, slot, lev = footprint_cells(env, pos, actions)
    fc = jnp.clip(foot, 0, env.g - 1)
    pdc = pd[lev[:, :, None, None], fc[..., 0], fc[..., 1]]
    wt = w.astype(jnp.float32) * slot[:, :, None, :]                    # [N,A,R,9]
    return (belief[fc[..., 0], fc[..., 1]] * pdc * wt).sum((2, 3))     # [N,A]


def potential(env: SearchEnvV2, belief, mask, pts):
    """Pull toward remaining belief mass: sum_b mass_b / (1 + d(p, b) / ring)."""
    g = env.g
    nb = -(-g // BLOCK)
    pad = nb * BLOCK - g
    b = jnp.pad(jnp.where(mask, belief, 0.0), ((0, pad), (0, pad)))
    mass = b.reshape(nb, BLOCK, nb, BLOCK).sum((1, 3)).ravel()
    cr, cc = jnp.meshgrid(jnp.arange(nb), jnp.arange(nb), indexing="ij")
    centers = (jnp.stack([cr.ravel(), cc.ravel()], -1) * BLOCK + BLOCK / 2).astype(jnp.float32)
    d = jnp.sqrt(((pts[..., None, :].astype(jnp.float32) - centers) ** 2).sum(-1))
    return (mass / (1.0 + d / env.r)).sum(-1)


def _end_ok(env: SearchEnvV2, mask, ends):
    g = env.g
    dil = jax.lax.reduce_window(mask.astype(jnp.int32), 0, jax.lax.max, (3, 3), (1, 1), "SAME") > 0
    inb = ((ends >= 0) & (ends < g)).all(-1)
    ec = jnp.clip(ends, 0, g - 1)
    return dil[ec[..., 0], ec[..., 1]] & inb


def _ends(env: SearchEnvV2, pos, actions):
    """Absolute end cell of each action: [N, A, 2] when actions is [N, A]."""
    lev, base = actions // env.a0, actions % env.a0
    return pos[:, None, :] + env.offsets[lev, base, -1]


def action_gain(env: SearchEnvV2, state):
    """[N, 66] one-step expected detections per action, scaled to 0..1 per drone.

    Network input feature (milestone 4): the known model's immediate value of
    every action, divided by the best legal action's value, so the network only
    has to learn the longer-horizon correction on top of greedy.
    """
    n, A = env.n, env.n_actions
    legal = env.action_mask(state)
    acts = jnp.broadcast_to(jnp.arange(A)[None, :], (n, A))
    v = jnp.where(legal, seg_value(env, state.belief, state.pd, state.pos, acts), 0.0)
    return v / jnp.maximum(v.max(-1, keepdims=True), 1e-9)


def greedy(env: SearchEnvV2, state, key, pull: float = 1e-3):
    """Highest immediate expected detections over both levels (plan section 8.1)."""
    n, A = env.n, env.n_actions
    legal = env.action_mask(state)
    acts = jnp.broadcast_to(jnp.arange(A)[None, :], (n, A))
    v = seg_value(env, state.belief, state.pd, state.pos, acts)          # [N,A]
    phi = potential(env, state.belief, state.mask, _ends(env, state.pos, acts))
    score = v + pull * phi + 1e-6 * jax.random.uniform(key, (n, A))
    return jnp.argmax(jnp.where(legal, score, NEG), -1)


def lookahead(env: SearchEnvV2, state, key, gamma: float = 0.5, pull: float = 0.05):
    """Two-segment look-ahead over both levels.

    The second segment starts at the end of the first, at either level. Cells the
    first segment looked at keep less belief, so the pair is not double counted.
    Drones plan in turn on a shared belief copy, like version 1.
    """
    n, A0, g = env.n, env.a0, env.g
    A = env.n_actions
    legal = env.action_mask(state)
    keys = jax.random.split(key, n)
    acts = []
    belief = state.belief
    lev_all = jnp.arange(A) // A0
    base_all = jnp.arange(A) % A0
    for i in range(n):
        p0 = state.pos[i]
        # first segment, every action
        f1 = env.foot[lev_all, base_all] + p0                            # [A,R,9,2]
        s1 = env.slot_table                                                # [A,9]
        v1 = env.valid[lev_all, base_all]                               # [A,R]
        inb1 = ((f1 >= 0) & (f1 < g)).all(-1)                            # [A,R,9]
        w1 = (v1[:, :, None] & inb1).astype(jnp.float32) * s1[:, None, :]
        c1 = jnp.clip(f1, 0, g - 1)
        pd1 = state.pd[lev_all[:, None, None], c1[..., 0], c1[..., 1]]
        val1 = (belief[c1[..., 0], c1[..., 1]] * pd1 * w1).sum((1, 2))  # [A]

        # per-cell belief remaining after each first segment. One scatter per
        # action (vmapped), which keeps the index and update layouts matching
        # without a manual transpose.
        def _left(c1a, w1a, pd1a):
            flat = c1a[..., 0] * g + c1a[..., 1]                      # [R,9]
            ww = w1a.ravel()                                          # [R*9]
            c = jnp.zeros(g * g).at[flat.ravel()].add(ww)
            p = jnp.zeros(g * g).at[flat.ravel()].add(ww * pd1a.ravel())
            eff = jnp.where(c > 0, p / jnp.maximum(c, 1e-6), 0.0)
            return belief * jnp.exp(-c * (1.0 - eff)).reshape(g, g)    # [G,G]

        left1 = jax.vmap(_left)(c1, w1, pd1)                          # [A,G,G]

        # second segment from every first-segment end, every action
        ends1 = p0 + env.offsets[lev_all, base_all, -1]                 # [A,2]
        f2 = ends1[:, None, None, :] + env.foot[lev_all, base_all][None]  # [A,A,R,9,2]
        s2 = env.slot_table                                                # [A,9]
        v2 = env.valid[lev_all, base_all]
        inb2 = ((f2 >= 0) & (f2 < g)).all(-1)
        w2 = (v2[:, None, :, None] & inb2).astype(jnp.float32) * s2[None, :, None, :]
        c2 = jnp.clip(f2, 0, g - 1)
        pd2 = state.pd[lev_all[None, :, None, None], c2[..., 0], c2[..., 1]]
        # gather left1 at the second segment's cells for each first action: [A,A,R,9]
        b2 = left1[jnp.arange(A)[:, None, None, None],
                   c2[..., 0], c2[..., 1]]                            # [A,A,R,9]
        val2 = (b2 * pd2 * w2).sum((2, 3))                               # [A,A]
        ok2 = _end_ok(env, state.mask, ends1[:, None, :] + env.offsets[lev_all, base_all, -1][None])
        phi2 = potential(env, belief, state.mask,
                         ends1[:, None, :] + env.offsets[lev_all, base_all, -1][None])
        tot = val1[:, None] + gamma * val2 + pull * phi2 / env.r
        tot = jnp.where(ok2, tot, NEG)
        tot = tot + 1e-7 * jax.random.uniform(keys[i], (A, A))
        best = tot.max(-1)
        a = jnp.argmax(jnp.where(legal[i], best, NEG))
        acts.append(a)
        # commit this drone's first segment to the shared belief copy: the belief
        # left at each cell after a looks is exactly left1[a] computed above.
        on = (state.status[i] == ACTIVE) & (a != env.hold) & (a != env.hold + A0)
        belief = jnp.where(on, left1[a], belief)
    return jnp.stack(acts)


def lawnmower(env: SearchEnvV2, state, key, level: int = LOW):
    """Boustrophedon rows at one level inside per-drone column strips.

    Mirrors the version 1 baseline (which works well) on top of the v2 action
    set: the searchable columns are split between the active drones, and each
    drone walks its strip toward the next unlooked cell in serpentine order,
    scoring the unlooked cells along the whole segment path first and the
    distance to that target second. Falls back to the whole area when the strip
    is swept, so a drone never idles.
    """
    n, g, A0 = env.n, env.g, env.a0
    legal = env.action_mask(state)
    active = state.status == ACTIVE
    rank = jnp.cumsum(active) - 1
    n_act = jnp.maximum(active.sum(), 1)
    cols = jnp.where(state.mask.any(0), jnp.arange(g), -1)
    cmin = jnp.min(jnp.where(cols >= 0, cols, g))
    cmax = jnp.max(cols)
    width = (cmax - cmin + 1) / n_act
    rr, cc = jnp.meshgrid(jnp.arange(g), jnp.arange(g), indexing="ij")
    order = rr * g + jnp.where(rr % 2 == 0, cc, g - 1 - cc)
    unlooked = state.mask & (state.looks == 0)
    lev = jnp.full((A0,), level, jnp.int32)
    base = jnp.arange(A0)
    acts = []
    for i in range(n):
        lo = cmin + rank[i] * width
        strip = (cc >= lo) & (cc < lo + width)
        todo = unlooked & strip
        todo = jnp.where(todo.any(), todo, unlooked)
        idx = jnp.argmin(jnp.where(todo, order, 10**9).ravel())
        tgt = jnp.array([idx // g, idx % g])
        p = state.pos[i]
        # segment path: the ring of cells from p to the action end [A0,R,2]
        path = env.offsets[lev, base] + p[None, None, :]                 # [A0,R,2]
        inb = ((path >= 0) & (path < g)).all(-1)
        w = (env.valid[lev, base] & inb).astype(jnp.float32)             # [A0,R]
        pc = jnp.clip(path, 0, g - 1)
        gain = (todo[pc[..., 0], pc[..., 1]] * w).sum(-1)                # [A0]
        ends = path[:, -1, :]                                            # [A0,2]
        dist = jnp.abs(ends - tgt[None, :]).max(-1).astype(jnp.float32)
        score = 10.0 * gain - dist
        score = score.at[env.hold].set(NEG)                              # never hold while sweeping
        ok = legal[i, jnp.arange(A0) + level * A0]
        a = jnp.argmax(jnp.where(ok, score, NEG))
        acts.append(jnp.where(active[i], a + level * A0, env.hold))
    return jnp.stack(acts)


POLICIES_V2 = {"lawnmower": lawnmower, "greedy": greedy, "planner": lookahead}