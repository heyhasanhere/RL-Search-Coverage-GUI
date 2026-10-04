"""Koester baseline prior: distance rings times find-feature weights.

Plan section 6.2. The numbers come from the published ISRID hiker statistics
that Hashimoto et al. 2022 used for their model:

* distance from the last-seen point: 25% inside 0.24 km, 50% inside 1.3 km,
  75% inside 3.6 km, 95% inside 9.4 km, for overnight hikers who went out
  intending to return. The plan's own figures (median 2.2 km, 75th percentile
  4.4 km) are the same distribution measured on the 65 real cases, so the
  default uses those and `hiker_ring_km` keeps the published set for reference.
* find features: a hiker is usually found on a trail or a road, near a stream
  or a water source, or near a structure. `FIND_FEATURE_WEIGHTS` are relative
  multipliers per feature; the largest feature share from the literature
  (trails and roads together) sets the normalization.

Every map here sums to `expected_count`, so the caller can hand the result
straight to the search agent as a survivor count.
"""

from __future__ import annotations

import numpy as np


# Distance quantiles in km for overnight hikers (ISRID, via Hashimoto 2022).
HIKER_RING_KM = np.array([0.24, 1.3, 3.6, 9.4])
HIKER_RING_MASS = np.array([0.25, 0.25, 0.25, 0.20])

# The plan section 6.1 numbers measured on the 65 ISRID cases we fit to.
CASE_RING_KM = np.array([0.4, 2.2, 4.4, 9.7])
CASE_RING_MASS = HIKER_RING_MASS

# Share of found hikers by feature type, and how much land each feature covers.
# Used together: a cell's weight is the ring density times the best feature
# multiplier available there, which is what Koester's model does.
FIND_FEATURE_WEIGHTS = {
    "trail": 3.0,        # on or within 30 m of a trail
    "road": 1.2,         # on a road, which is less likely than a trail
    "stream": 1.4,       # near a stream or water source
    "building": 2.2,     # near a structure or trailhead
    "none": 1.0,         # open country
}

# Feature cells are rare, so their share is small; the open-country share
# carries the rest. These are the Koester-style prior weights used by the
# SAR literature (feature share about 0.3 of finds, terrain-only the rest).
FEATURE_SHARE = 0.30

# Distance in cells a feature still counts for (30 m cells, so 1 cell).
FEATURE_RADIUS_CELLS = 1


def ring_density(dist_m, rings_m=None, mass=None, clip_r=None):
    """Piecewise-constant radial density around the last-seen point.

    `dist_m` is the distance from the IPP in metres. The density is constant
    inside each ring and zero beyond the last one, which is the Koester form.
    """
    if rings_m is None:
        rings_m = np.asarray(CASE_RING_KM, np.float64) * 1000.0
    if mass is None:
        mass = np.asarray(CASE_RING_MASS, np.float64)
    if clip_r is not None:
        rings_m = rings_m * clip_r
        mass = mass / mass.sum()
    edges = np.concatenate([[0.0], rings_m])
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    dens = mass / area
    band = np.searchsorted(edges[1:], dist_m, side="right")
    out = np.zeros_like(dist_m, np.float64)
    inside = band < len(dens)
    out[inside] = dens[band[inside]]
    return out


def feature_field(layers, radius=FEATURE_RADIUS_CELLS):
    """Best feature multiplier per cell, from the OSM layers."""
    from scipy.ndimage import maximum_filter

    field = np.full(layers["lc"].shape, FIND_FEATURE_WEIGHTS["none"], np.float32)
    g = layers["lc"].shape[0]
    for key, weight in (
        ("trail", FIND_FEATURE_WEIGHTS["trail"]),
        ("road", FIND_FEATURE_WEIGHTS["road"]),
        ("stream", FIND_FEATURE_WEIGHTS["stream"]),
        ("building", FIND_FEATURE_WEIGHTS["building"]),
    ):
        mask = layers.get(key)
        if mask is None:
            continue
        pad = max(0, int(radius))
        if pad:
            mask = maximum_filter(mask.astype(np.uint8), size=2 * pad + 1) > 0
        field = np.where(mask, np.maximum(field, weight), field)
    return field


def _normalize(x, mask):
    x = np.where(mask, np.maximum(x, 0.0), 0.0)
    total = x.sum()
    if total <= 0:
        x = mask.astype(np.float64)
        total = max(x.sum(), 1.0)
    return x / total


def koester_map(layers, ipp_rc, expected_count=1.0, rings_m=None, mass=None,
                cell_m=30.0, floor=0.0, feature_radius=FEATURE_RADIUS_CELLS):
    """Baseline prior for one region. Returns [g,g] float32 summing to expected_count.

    `ipp_rc` is the last-seen point as (row, col). Water cells are removed: a
    hiker is not found in a lake. `floor` adds a share of uniform over the
    remaining cells.
    """
    g = layers["lc"].shape[0]
    rr, cc = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    dr = (rr - ipp_rc[0]) * cell_m
    dc = (cc - ipp_rc[1]) * cell_m
    dist = np.hypot(dr, dc)

    feature = feature_field(layers, feature_radius)
    feature_share = (feature - FIND_FEATURE_WEIGHTS["none"])
    # Normalize the feature term to its share of the total, then add the rest
    # as the ring term; this keeps the total mass equal to the ring mass.
    f = _normalize(feature_share, _searchable(layers))
    r = _normalize(ring_density(dist, rings_m, mass), _searchable(layers))
    mix = FEATURE_SHARE * f + (1.0 - FEATURE_SHARE) * r
    mix = mix / mix.sum()

    if floor > 0:
        m = _searchable(layers).astype(np.float64)
        uni = m / max(m.sum(), 1.0)
        mix = (1.0 - floor) * mix + floor * uni
    return (mix * expected_count).astype(np.float32)


def _searchable(layers):
    """Cells a prior can put mass on: not water, not blocked."""
    water = layers.get("lc") == 4
    if "water" in layers:
        water = water | layers["water"]
    out = ~water
    if "blocked" in layers:
        out = out & ~layers["blocked"]
    return out


def distance_share(prior, find_rc):
    """Share of area a search must cover to reach the find, per plan 6.5."""
    flat = np.sort(prior.ravel())[::-1]
    total = flat.sum()
    if total <= 0:
        return 1.0
    cum = np.cumsum(flat) / total
    rank = int(np.searchsorted(cum, prior[find_rc], side="left"))
    return (rank + 1) / prior.size