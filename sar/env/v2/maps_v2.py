"""Procedural polygons, priors, and terrain-class priors for env v2 (plan sections 7.1, 7.4).

Version 2 adds two things to `sar.env.maps`:
  - land cover, canopy height, DEM and floor maps per cell, so Pd and the flight
    floor are per cell (plan sections 7.1, 9.1)
  - a prior bank loader for `data/regions/bank_{train,heldout}.npz`

Until the real prior worker writes `sar/prior/**` and the bank files, a synthetic
land-cover/canopy generator stands behind the same array interface, so training is
not blocked. The bank loader picks the real files up as soon as they appear.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sar.config_v2 import N_CANOPY

# Reuse the version 1 procedural priors so v1 and v2 are directly comparable.
from sar.env.maps import (  # noqa: F401
    _blob_prior,
    _line_prior,
    _ring_prior,
    cell_coords,
    sample_polygon,
    _random_point_in_mask,
)


def norm_in_mask(x, mask):
    x = jnp.where(mask, x, 0.0)
    return x / jnp.maximum(x.sum(), 1e-12)


def sample_prior_v2(key, mask, kind):
    """Prior intensity map (sums to 1 inside the mask). Kind 0-3 as in v1."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    ring = norm_in_mask(_ring_prior(k1, mask), mask)
    lines = norm_in_mask(_line_prior(k2, mask), mask)
    blobs = norm_in_mask(_blob_prior(k3, mask), mask)
    shaped = jnp.select([kind == 0, kind == 1, kind == 2], [ring, lines, 0.5 * ring + 0.5 * lines], blobs)
    floor_w = jax.random.uniform(k4, minval=0.05, maxval=0.3)
    return norm_in_mask((1 - floor_w) * shaped + floor_w * norm_in_mask(jnp.ones_like(ring), mask), mask)


# --------------------------------------------------------------------- terrain
# Land cover order: 0 water, 1 grass, 2 forest, 3 cropland, 4 built (config_v2)
# Canopy order: 0 none, 1 sparse, 2 dense, 3 tall (config_v2)

def _fbm(rng, g, cells):
    """Single-octave value-noise field bilinearly upsampled to g x g."""
    small = max(2, min(g, cells))
    base = rng.random((small, small)).astype(np.float32)
    idx = np.linspace(0, small - 1, g)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, small - 1)
    w = (idx - i0).astype(np.float32)
    w2 = w[:, None] * w[None, :]
    return (base[i0[:, None], i0[None, :]] * (1 - w2)
            + base[i1[:, None], i0[None, :]] * (w[:, None] * (1 - w[None, :]))
            + base[i0[:, None], i1[None, :]] * ((1 - w[:, None]) * w[None, :])
            + base[i1[:, None], i1[None, :]] * w2)


def _normalize(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / max(hi - lo, 1e-6)


def synthetic_terrain_np(key: int, g: int, seed: int = 0):
    """Synthetic land cover, canopy class, DEM and floor for a g x g grid.

    Stands in for the real prior worker's layers until `data/regions/bank_*.npz`
    exists. Class fractions are set by percentile thresholds on independent
    normalized fields, so the mix is stable across seeds (roughly water 5%,
    built 6%, forest 40%, crop 12% of the remainder grass). Returns a dict of
    numpy arrays: landcover [G,G], canopy [G,G], dem_m [G,G], floor_m [G,G].
    """
    rng = np.random.default_rng(seed ^ (key * 0x9E3779B9))
    water_f = _normalize(_fbm(rng, g, g // 2))
    forest_f = _normalize(_fbm(rng, g, g // 3))
    built_f = _normalize(_fbm(rng, g, g // 6))
    elev = _normalize(_fbm(rng, g, g // 4))

    water = water_f > np.percentile(water_f, 95)             # ~5% streams and lakes
    built = (~water) & (built_f > np.percentile(built_f, 94))  # ~6% clusters
    forest = (~water) & (~built) & (forest_f > np.percentile(forest_f, 62))  # ~38%
    crop = (~water) & (~built) & (~forest) & (rng.random((g, g)) < 0.30)      # ~30% of grass
    landcover = np.where(water, 0, np.where(built, 4, np.where(forest, 2, np.where(crop, 3, 1))))

    # Canopy height class follows the forest mask; built cells get tall (roofs).
    canopy_m = np.where(landcover == 2, rng.uniform(5, 30, (g, g)), 0.0).astype(np.float32)
    canopy = np.zeros((g, g), np.int32)
    canopy[landcover == 2] = np.digitize(canopy_m[landcover == 2], [8.0, 18.0]).astype(np.int32)  # 1,2,3
    canopy[landcover == 4] = rng.integers(2, 4, built.sum())  # buildings: tall canopy class
    canopy = np.clip(canopy, 0, N_CANOPY - 1)

    # DEM and floor: ground elevation + obstacle height + clearance.
    dem_m = (elev * 60.0).astype(np.float32)
    obstacle_m = np.where(landcover == 4, rng.uniform(6, 35, (g, g)),
                          np.where(canopy > 0, canopy_m, 0.0)).astype(np.float32)
    floor_m = dem_m + obstacle_m + 10.0
    return {"landcover": landcover.astype(np.int32), "canopy": canopy.astype(np.int32),
            "dem_m": dem_m, "floor_m": floor_m}


# --------------------------------------------------------------- prior bank
BANK_DIR = Path("data/regions")


BANK_ENV_KEYS = ("landcover", "canopy", "dem_m", "floor_m", "prior")


def load_prior_bank(split: str = "train", bank_dir: Path | str = BANK_DIR, grid: int | None = None):
    """Load `bank_{split}.npz` if present, else None (use procedural/synthetic).

    The file (written by `sar.prior.service.write_bank`) stacks every region on
    a leading axis: `landcover`, `canopy`, `dem_m`, `floor_m`, `prior` [N,G,G]
    plus metadata (`name`, `ipp_rc`, `lat`, `lon`, ...). With `grid` smaller
    than G, each map is cropped to a `grid` square centred on the region's
    last-seen point (kept inside the map) and the prior is renormalized, so
    the curriculum grids 32/64/128 use the same bank.
    """
    path = Path(bank_dir) / f"bank_{split}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    g_full = out["landcover"].shape[-1]
    if grid is None or int(grid) >= g_full:
        return out
    g = int(grid)
    n = out["landcover"].shape[0]
    ipp = out.get("ipp_rc", np.full((n, 2), g_full // 2, np.int32))
    r0 = np.clip(ipp[:, 0] - g // 2, 0, g_full - g)
    c0 = np.clip(ipp[:, 1] - g // 2, 0, g_full - g)
    for k, v in list(out.items()):
        if v.ndim == 3 and v.shape[1:] == (g_full, g_full):
            out[k] = np.stack([v[i, r0[i]:r0[i] + g, c0[i]:c0[i] + g] for i in range(n)])
    if "prior" in out:
        tot = out["prior"].reshape(n, -1).sum(1)
        out["prior"] = (out["prior"] / np.maximum(tot, 1e-12)[:, None, None]).astype(np.float32)
    if "ipp_rc" in out:
        out["ipp_rc"] = np.stack([ipp[:, 0] - r0, ipp[:, 1] - c0], 1).astype(np.int32)
    return out
