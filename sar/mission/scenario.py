"""Simulated survivors and Tier 0-style maps for orchestrator runs (plan 10.1).

Survivor positions are sampled from a reward map that mixes the prior with a
uniform map, as in training, so the prior is informative but not exact.
"""

import jax
import numpy as np

from sar.mission.geo import AreaGrid


def sample_survivors(grid: AreaGrid, prior, n: int, rng: np.random.Generator, mix: float = 0.25):
    m = grid.mask
    uni = m / m.sum()
    p = ((1 - mix) * prior + mix * uni).ravel()
    p = p / p.sum()
    idx = rng.choice(p.size, size=n, replace=True, p=p)
    r, c = idx // grid.g, idx % grid.g
    e, nn = grid.cell_center(r, c)
    jitter = rng.uniform(-0.45, 0.45, (n, 2)) * grid.cell
    return np.stack([e + jitter[:, 0], nn + jitter[:, 1]], -1)


def tier_map(seed: int, g: int, kinds=(3,)):
    """Polygon mask, prior (sums to 1), and survivor count from the JAX env reset."""
    from sar.eval.tier0 import make_eval_env
    env = make_eval_env(g, 4, kinds, False)
    s, _ = env.reset(jax.random.PRNGKey(seed))
    mask = np.asarray(s.mask)
    prior = np.asarray(s.belief) / float(s.belief.sum())
    n_surv = max(1, int(round(float(s.belief0))))
    return mask, prior, n_surv
