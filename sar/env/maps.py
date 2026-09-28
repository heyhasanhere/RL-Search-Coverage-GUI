"""Procedural polygons and prior maps in JAX (plan sections 5.1, 5.2, 5.7)."""

import jax
import jax.numpy as jnp
import numpy as np


def cell_coords(g: int):
    """Return (rows, cols) float grids of cell centers."""
    r, c = jnp.meshgrid(jnp.arange(g), jnp.arange(g), indexing="ij")
    return r.astype(jnp.float32), c.astype(jnp.float32)


def sample_polygon(key, g: int, extent_min: float, extent_max: float, n_vert: int = 8):
    """Star-shaped random polygon rasterized onto a g x g grid. Returns bool mask."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    extent = jax.random.uniform(k1, minval=extent_min, maxval=extent_max)
    rad = extent / 2.0
    lo = rad + 1.0
    hi = g - rad - 2.0
    center = jax.random.uniform(k2, (2,), minval=lo, maxval=jnp.maximum(hi, lo + 1e-3))
    radii = rad * jax.random.uniform(k3, (n_vert,), minval=0.55, maxval=1.0)
    jitter = jax.random.uniform(k4, (n_vert,), minval=-0.3, maxval=0.3)
    angles = (jnp.arange(n_vert) + jitter) * (2 * jnp.pi / n_vert)

    rr, cc = cell_coords(g)
    dy, dx = rr - center[0], cc - center[1]
    theta = jnp.mod(jnp.arctan2(dy, dx), 2 * jnp.pi)
    dist = jnp.sqrt(dx**2 + dy**2)
    # Periodic linear interpolation of radius over angle
    ang_ext = jnp.concatenate([angles, angles[:1] + 2 * jnp.pi])
    rad_ext = jnp.concatenate([radii, radii[:1]])
    theta_w = jnp.where(theta < angles[0], theta + 2 * jnp.pi, theta)
    r_at = jnp.interp(theta_w, ang_ext, rad_ext)
    mask = dist <= r_at
    # Guarantee the center cell is inside
    ci = jnp.clip(jnp.round(center).astype(jnp.int32), 0, g - 1)
    return mask.at[ci[0], ci[1]].set(True)


def _random_point_in_mask(key, mask):
    logits = jnp.where(mask.ravel(), 0.0, -jnp.inf)
    idx = jax.random.categorical(key, logits)
    g = mask.shape[0]
    return jnp.array([idx // g, idx % g], dtype=jnp.float32)


def _ring_prior(key, mask):
    """Koester-style distance rings around a last-known point."""
    g = mask.shape[0]
    k1, k2 = jax.random.split(key)
    lkp = _random_point_in_mask(k1, mask)
    scale = jax.random.uniform(k2, minval=3.0, maxval=g / 4)
    q = scale * jnp.array([0.4, 1.0, 1.8, 3.2])  # 25/50/75/95% distances
    rr, cc = cell_coords(g)
    d = jnp.sqrt((rr - lkp[0]) ** 2 + (cc - lkp[1]) ** 2)
    edges = jnp.concatenate([jnp.zeros(1), q])
    mass = jnp.array([0.25, 0.25, 0.25, 0.20])
    area = jnp.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    dens = mass / area
    band = jnp.searchsorted(q, d)
    outer = 0.05 / (g * g)
    return jnp.where(band < 4, dens[jnp.minimum(band, 3)], outer)


def _line_prior(key, mask, n_lines: int = 3):
    """Roads and rivers as gently curved lines with a Gaussian falloff."""
    g = mask.shape[0]
    rr, cc = cell_coords(g)
    keys = jax.random.split(key, n_lines)

    def one(k):
        ka, kb, kc, kd, ke = jax.random.split(k, 5)
        p0 = jax.random.uniform(ka, (2,), maxval=g)
        ang = jax.random.uniform(kb, maxval=jnp.pi)
        sigma = jax.random.uniform(kc, minval=1.0, maxval=3.0)
        amp = jax.random.uniform(kd, minval=0.0, maxval=6.0)
        w = jax.random.uniform(ke, minval=0.3, maxval=1.0)
        nrm = jnp.array([-jnp.sin(ang), jnp.cos(ang)])
        tan = jnp.array([jnp.cos(ang), jnp.sin(ang)])
        along = (rr - p0[0]) * tan[0] + (cc - p0[1]) * tan[1]
        perp = (rr - p0[0]) * nrm[0] + (cc - p0[1]) * nrm[1]
        perp = perp - amp * jnp.sin(along / 12.0)
        return w * jnp.exp(-(perp**2) / (2 * sigma**2))

    return jax.vmap(one)(keys).sum(0)


def _blob_prior(key, mask, n_blobs: int = 4):
    """Gaussian clusters. Held out from training for generalization tests."""
    g = mask.shape[0]
    rr, cc = cell_coords(g)
    keys = jax.random.split(key, n_blobs)

    def one(k):
        ka, kb, kc = jax.random.split(k, 3)
        c = _random_point_in_mask(ka, mask)
        s = jax.random.uniform(kb, minval=2.0, maxval=g / 8)
        w = jax.random.uniform(kc, minval=0.2, maxval=1.0)
        return w * jnp.exp(-((rr - c[0]) ** 2 + (cc - c[1]) ** 2) / (2 * s**2)) / s**2

    return jax.vmap(one)(keys).sum(0)


def sample_prior(key, mask, kind):
    """Prior intensity map (sums to 1 inside the mask) for prior type `kind`."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    ring = _ring_prior(k1, mask)
    lines = _line_prior(k2, mask)
    blobs = _blob_prior(k3, mask)

    def norm(x):
        x = jnp.where(mask, x, 0.0)
        return x / jnp.maximum(x.sum(), 1e-12)

    ring, lines, blobs = norm(ring), norm(lines), norm(blobs)
    shaped = jnp.select(
        [kind == 0, kind == 1, kind == 2],
        [ring, lines, 0.5 * ring + 0.5 * lines],
        blobs,
    )
    floor_w = jax.random.uniform(k4, minval=0.05, maxval=0.3)
    uniform = norm(jnp.ones_like(ring))
    return norm((1 - floor_w) * shaped + floor_w * uniform)


REAL_KIND = 4  # prior type drawn from a real-terrain bank (region bank and demo world)


def load_real_bank(paths, g: int, n_crops: int = 8, seed: int = 0):
    """Stack of [M,g,g] real priors cropped from region npz files (plan 21.1 step 2).

    Each region gives n_crops crops of side g with the IPP inside the middle half of the
    crop, so the drawn search area covers the likely find cells as in a real mission.
    A region smaller than g is padded with zeros (outside any polygon of that size).
    """
    rng = np.random.default_rng(seed)
    out = []
    for p in paths:
        d = np.load(p)
        pri = d["prior"].astype(np.float32)
        ipp = d["ipp_rc"]
        if pri.ndim == 2:
            pri, ipp = pri[None], ipp[None]
        for k in range(len(pri)):
            m, s = pri[k], pri[k].shape[0]
            if s < g:
                pad = np.zeros((g, g), np.float32)
                o = (g - s) // 2
                pad[o:o + s, o:o + s] = m
                out += [pad] * n_crops
                continue
            for _ in range(n_crops):
                r0, c0 = (int(np.clip(ipp[k][i] - rng.integers(g // 4, 3 * g // 4 + 1), 0, s - g)) for i in (0, 1))
                out.append(m[r0:r0 + g, c0:c0 + g])
    return np.stack(out)


def sample_real_prior(key, mask, bank):
    """Random bank map with a random rotation and flip, inside the mask, with a uniform floor."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    m = bank[jax.random.randint(k1, (), 0, bank.shape[0])]
    rot = jax.random.randint(k2, (), 0, 4)
    m = jax.lax.switch(rot, [lambda x, r=r: jnp.rot90(x, r) for r in range(4)], m)
    m = jnp.where(jax.random.bernoulli(k3), m[:, ::-1], m)

    def norm(x):
        x = jnp.where(mask, x, 0.0)
        return x / jnp.maximum(x.sum(), 1e-12)

    floor_w = jax.random.uniform(k4, minval=0.05, maxval=0.3)
    return norm((1 - floor_w) * norm(m) + floor_w * norm(jnp.ones_like(m)))
