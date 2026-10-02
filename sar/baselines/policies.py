"""Lawnmower, greedy, and look-ahead planner baselines in JAX (plan section 6).

Each policy maps (env, state, key) -> actions [N] for one env; batch with vmap.
"""

import jax
import jax.numpy as jnp

from sar.env.grid_env import ACTIVE, SearchEnv

NEG = -1e9
BLOCK = 8          # block size for the long-range pull potential


def _seg_value(env: SearchEnv, belief, pd, paths, valid):
    """Expected detections of each segment: paths [..., R, 2], valid [..., R]."""
    g = env.g
    pc = jnp.clip(paths, 0, g - 1)
    inb = ((paths >= 0) & (paths < g)).all(-1)
    v = belief[pc[..., 0], pc[..., 1]] * pd * valid * inb
    return v.sum(-1)


def _potential(env: SearchEnv, belief, mask, pts):
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


def _end_ok(env: SearchEnv, mask, ends):
    g = env.g
    dil = jax.lax.reduce_window(mask.astype(jnp.int32), 0, jax.lax.max, (3, 3), (1, 1), "SAME") > 0
    inb = ((ends >= 0) & (ends < g)).all(-1)
    ec = jnp.clip(ends, 0, g - 1)
    return dil[ec[..., 0], ec[..., 1]] & inb


def greedy(env: SearchEnv, state, key, pull: float = 1e-3):
    legal = env.action_mask(state)
    paths = env.paths(state.pos)                                        # [N,A,R,2]
    v = _seg_value(env, state.belief, state.pd, paths, env.valid[None])
    phi = _potential(env, state.belief, state.mask, paths[:, :, -1])
    score = v + pull * phi + 1e-6 * jax.random.uniform(key, v.shape)
    return jnp.argmax(jnp.where(legal, score, NEG), -1)


def lookahead(env: SearchEnv, state, key, gamma: float = 0.5, pull: float = 0.05):
    """Two-segment look-ahead; drones plan in turn on a shared belief copy."""
    n, A, R, g = env.n, env.n_actions, env.r, env.g
    legal = env.action_mask(state)
    off, valid = env.offsets, env.valid
    belief = state.belief
    pd = state.pd
    acts = []
    for i in range(n):
        p0 = state.pos[i]
        p1 = p0 + off                                                    # [A,R,2]
        e1 = p1[:, -1]                                                   # [A,2]
        p2 = e1[:, None, None, :] + off[None]                            # [A,A,R,2]
        v1 = _seg_value(env, belief, pd, p1, valid)                      # [A]
        # second segment value, discounting cells already looked at by segment one
        same = (p2[:, :, :, None, :] == p1[:, None, None, :, :]).all(-1)  # [A,A,R,R]
        rep = (same & valid[:, None, None, :]).any(-1)                   # [A,A,R]
        pc = jnp.clip(p2, 0, g - 1)
        inb = ((p2 >= 0) & (p2 < g)).all(-1)
        b2 = belief[pc[..., 0], pc[..., 1]] * jnp.where(rep, 1.0 - pd, 1.0)
        v2 = (b2 * pd * valid[None] * inb).sum(-1)                       # [A,A]
        ok2 = _end_ok(env, state.mask, p2[:, :, -1])
        phi = _potential(env, belief, state.mask, p2[:, :, -1])         # [A,A]
        tot = v1[:, None] + gamma * v2 + pull * pd * phi / env.r
        tot = jnp.where(ok2, tot, NEG)
        best = tot.max(-1) + 1e-7 * jax.random.uniform(jax.random.fold_in(key, i), (A,))
        a = jnp.argmax(jnp.where(legal[i], best, NEG))
        acts.append(a)
        # commit this drone's first segment to the shared belief copy
        on = (state.status[i] == ACTIVE)
        cells = jnp.clip(p1[a], 0, g - 1)
        cnt = jnp.zeros((g, g)).at[cells[:, 0], cells[:, 1]].add(valid[a] * on)
        belief = belief * (1.0 - pd) ** cnt
    return jnp.stack(acts)


def lawnmower(env: SearchEnv, state, key):
    """Boustrophedon rows inside per-drone column strips; re-split when the fleet changes."""
    n, g = env.n, env.g
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
    paths = env.paths(state.pos)                                        # [N,A,R,2]
    acts = []
    for i in range(n):
        lo = cmin + rank[i] * width
        strip = (cc >= lo) & (cc < lo + width)
        todo = unlooked & strip
        todo = jnp.where(todo.any(), todo, unlooked)
        idx = jnp.argmin(jnp.where(todo, order, 10**9).ravel())
        tgt = jnp.array([idx // g, idx % g])
        pc = jnp.clip(paths[i], 0, g - 1)
        gain = (todo[pc[..., 0], pc[..., 1]] * env.valid).sum(-1)
        dist = jnp.abs(paths[i, :, -1] - tgt).max(-1)
        score = 10.0 * gain - dist
        acts.append(jnp.argmax(jnp.where(legal[i], score, NEG)))
    return jnp.stack(acts)


POLICIES = {"lawnmower": lawnmower, "greedy": greedy, "planner": lookahead}
