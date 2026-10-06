"""Flood prior and flood survivor sampler (plan sections 5, 6.4, 12.3 and 13.3).

The flood prior has no case data to fit (plan 6.4), so it is a hand-set mix
of three components inside the flood extent plus a uniform floor:

| Key | Who | Spatial weight |
|---|---|---|
| `roofs` | People in flooded buildings whose roof is above water | People per building cell |
| `high_ground` | People who left flooded buildings for dry ground | Flooded-building people spread with a 90 m kernel over dry cells near the flood, islands doubled |
| `water` | People caught while driving or walking | Flooded cells next to a road, outside the normal river |
| `uniform` | A wrong map never removes the true location (plan 6.3) | Every cell in the area |

Water level
-----------
`water_level_m` is the flood stage in metres above the normal river level
(`river_level`, the 20th percentile of the DEM over the river cells plus
0.3 m). The UI sends it as `start_mission.water_level_m` (>= 0, default 2).

Flood extent: every cell below the water surface that connects (8-neighbour)
to the river. Low ground behind a levee that does not connect stays dry. Dry
blobs under 4 cells cut off by the flood are DEM bumps (trees, big roofs in
the surface model) and count as flooded with depth 0.

The survivor sampler for evaluation (plan 13.3) uses the same extent but
changes the component shares by a factor of 2 (roofs halved, water and high
ground doubled) and uses different spatial weights inside each component, so
a search method only wins by searching well, not by sharing the prior's
assumptions.
"""

from __future__ import annotations

import math
import os

import numpy as np
import scipy.ndimage as ndi

# Component shares of the prior. Hand-set (plan 6.4: no case data); the
# reasons are in docs/DECISIONS.md. They sum to 1 before the uniform floor.
SHARES = {"roofs": 0.60, "high_ground": 0.25, "water": 0.15}
UNIFORM_FLOOR = 0.05            # same floor as the wilderness blend (plan 6.3)

RIVER_OFFSET_M = 0.3            # river surface above the 20th percentile DEM of river cells
ROOF_FREEBOARD_M = 0.5          # a roof must stand this far above water to hold people
MIN_BUILDING_M = 3.0            # Isaac draws every building at least this tall
HIGH_SIGMA_CELLS = 3.0          # 90 m: how far people walk or wade to dry ground
HIGH_REACH_CELLS = 8            # dry cells further than 240 m from the flood get nothing
ISLAND_BONUS = 2.0              # dry islands inside the flood hold people who cannot walk out
ROAD_REACH_CELLS = 1            # water cells within one cell of a road
SHALLOW_M = 1.2                 # sampler: people standing in water stay visible above it
# Copernicus DEM is a surface model: a clump of trees or a large building
# lifts one or two cells above the flood. Dry blobs this small, cut off by the
# flood, are counted as flooded (depth 0), so their buildings still get roofs.
MIN_ISLAND_CELLS = 4

# Fallback occupancy when a layer set has no population grid: residents per
# square metre of building footprint. 2.5 people per house of about 150 m2
# footprint in an Australian country town (ABS 2021 average household 2.5).
PEOPLE_PER_M2 = 2.5 / 150.0

KINDS = ("roofs", "high_ground", "water")
DEFAULT_LAYERS = os.environ.get("SAR_FLOOD_LAYERS", "data/regions/demo_flood.npz")


# --------------------------------------------------------------- inputs

def river_seed(layers):
    """Cells that are water at normal level: OSM river and water bodies, WorldCover water."""
    seed = np.asarray(layers.get("river", layers["water"]), bool).copy()
    if "worldcover" in layers:
        seed |= np.asarray(layers["worldcover"]) == 80
    return seed


def river_level(layers):
    """Normal river surface in the DEM frame of `layers`."""
    if "river_level" in layers:
        return float(layers["river_level"])
    dem = np.asarray(layers["dem"], np.float64)
    seed = river_seed(layers)
    if not seed.any():
        return float(np.percentile(dem, 1))
    return float(np.percentile(dem[seed], 20)) + RIVER_OFFSET_M


def people(layers):
    """People per cell. GHSL population spread over buildings when present, else footprint area."""
    if "pop" in layers:
        return np.asarray(layers["pop"], np.float64)
    bld = np.asarray(layers["building"], bool)
    if "bldg_area" in layers:
        area = np.asarray(layers["bldg_area"], np.float64)
    else:
        cell = float(layers.get("cell_m", 30.0))
        area = bld * 0.25 * cell * cell          # a house covers about a quarter of a 30 m cell
    return area * PEOPLE_PER_M2


def flood_extent(layers, water_level_m):
    """(flooded bool map, depth map in m). Cells below the surface connected to the river."""
    dem = np.asarray(layers["dem"], np.float64)
    level = river_level(layers) + float(water_level_m)
    seed = river_seed(layers)
    below = (dem < level) | seed
    lab, _ = ndi.label(below, structure=np.ones((3, 3), bool))
    keep = np.unique(lab[seed & (lab > 0)])
    flooded = np.isin(lab, keep[keep > 0])
    dlab, n = ndi.label(~flooded, structure=np.ones((3, 3), bool))
    if n:
        sizes = np.bincount(dlab.ravel())
        edge = np.unique(np.concatenate([dlab[0], dlab[-1], dlab[:, 0], dlab[:, -1]]))
        small = (sizes < MIN_ISLAND_CELLS)
        small[0] = False
        small[edge] = False
        flooded |= small[dlab]
    depth = np.where(flooded, np.maximum(level - dem, 0.0), 0.0)
    return flooded, depth


def _islands(dry):
    """Dry components that do not touch the map border (surrounded by flood)."""
    lab, n = ndi.label(dry, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.zeros_like(dry)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return (lab > 0) & ~np.isin(lab, edge)


def _norm(a):
    s = float(a.sum())
    return a / s if s > 0 else np.zeros_like(a, np.float64)


# --------------------------------------------------------------- components

def flood_components(layers, water_level_m, variant="prior"):
    """Unit-mass spatial maps per component (each sums to 1, or 0 when empty).

    `variant="prior"` is the plan 6.4 model. `variant="sampler"` is the
    evaluation model: roofs weighted by building presence instead of people,
    a twice wider high-ground kernel without the island bonus, and water in
    any shallow flooded cell instead of only next to roads.
    """
    flooded, depth = flood_extent(layers, water_level_m)
    seed = river_seed(layers)
    bld = np.asarray(layers["building"], bool)
    bh = np.maximum(np.asarray(layers.get("bheight", np.where(bld, 6.0, 0.0)), np.float64),
                    MIN_BUILDING_M)
    pop = people(layers)
    fb = flooded & bld & ~seed
    roof_dry = fb & (bh > depth + ROOF_FREEBOARD_M)
    dry = ~flooded

    # Roofs: people of flooded buildings whose roof is still out of the water.
    if variant == "prior":
        roofs = np.where(roof_dry, pop, 0.0)
    else:
        roofs = roof_dry.astype(np.float64)

    # High ground: everyone from flooded buildings (submerged ones too) walks
    # or wades to nearby dry ground.
    src = np.where(fb, pop, 0.0)
    sigma = HIGH_SIGMA_CELLS * (1.0 if variant == "prior" else 2.0)
    spread = ndi.gaussian_filter(src, sigma, mode="constant")
    near = ndi.distance_transform_edt(~flooded) <= HIGH_REACH_CELLS * (1 if variant == "prior" else 2)
    high = np.where(dry & near, spread, 0.0)
    if variant == "prior":
        high = high * np.where(_islands(dry), ISLAND_BONUS, 1.0)

    # Water: people caught on roads, outside the normal river channel.
    wet = flooded & ~seed
    if variant == "prior":
        road = np.asarray(layers.get("road", np.zeros_like(bld)), bool)
        water = (wet & ndi.binary_dilation(road, iterations=ROAD_REACH_CELLS)).astype(np.float64)
    else:
        water = (wet & (depth <= SHALLOW_M)).astype(np.float64)
    return {"roofs": _norm(roofs), "high_ground": _norm(high), "water": _norm(water)}, flooded, depth


def _mix(comps, shares, area, floor):
    """Blend unit maps with shares; an empty component gives its share to the others."""
    live = {k: s for k, s in shares.items() if comps[k].sum() > 0}
    tot = sum(live.values())
    uni = _norm(area.astype(np.float64))
    out = {}
    for k in KINDS:
        w = (1.0 - floor) * live[k] / tot if k in live else 0.0
        out[k] = w * np.where(area, comps[k], 0.0)
    out["uniform"] = (floor if live else 1.0) * uni
    blend = sum(out.values())
    s = float(blend.sum())
    out = {k: v / s for k, v in out.items()}
    out["blend"] = sum(out[k] for k in (*KINDS, "uniform"))
    return out


def flood_prior(layers, water_level_m, area=None, shares=SHARES, floor=UNIFORM_FLOOR):
    """Plan 6.4 flood prior on the layer grid.

    Returns {"roofs", "high_ground", "water", "uniform", "blend"}; each map is
    float64 [H,W], the components are already scaled by their share, and
    "blend" (their sum) sums to 1 over `area` (default: every cell).
    """
    area = np.ones(np.shape(layers["dem"]), bool) if area is None else np.asarray(area, bool)
    comps, _, _ = flood_components(layers, water_level_m, "prior")
    return _mix(comps, shares, area, floor)


# --------------------------------------------------------------- evaluation sampler

def sampler_shares(factor=2.0, shares=SHARES):
    """Plan 13.3: roofs divided by `factor`, water and high ground multiplied by it."""
    raw = {"roofs": shares["roofs"] / factor, "high_ground": shares["high_ground"] * factor,
           "water": shares["water"] * factor}
    tot = sum(raw.values())
    return {k: v / tot for k, v in raw.items()}


def sampler_map(layers, water_level_m, factor=2.0, area=None):
    """The survivor model as maps: same keys as `flood_prior`, no uniform floor."""
    area = np.ones(np.shape(layers["dem"]), bool) if area is None else np.asarray(area, bool)
    comps, _, _ = flood_components(layers, water_level_m, "sampler")
    return _mix(comps, sampler_shares(factor), area, 0.0)


POSES = {"roofs": ("standing", "sitting"), "high_ground": ("standing", "sitting", "lying"),
         "water": ("standing",)}


def sample_flood_survivors(layers, water_level_m, n, rng, factor=2.0, area=None):
    """Flood test survivors (plan 12.3, 13.3) on the layer grid.

    Returns a list of dicts with the cell (`r`, `c`), the component (`kind`),
    and Isaac world coordinates in the frame of `sim/isaac/world.py`: `x` east
    and `y` north of the grid centre, `z` the standing height (roof top for
    roofs, None for ground), and a `pose`. Roof people stay inside the box
    Isaac draws for a building cell (0.6 x 0.5 of the cell, centred).
    """
    maps = sampler_map(layers, water_level_m, factor, area)
    dem = np.asarray(layers["dem"], np.float64)
    med = float(np.median(dem))
    bh = np.maximum(np.asarray(layers.get("bheight", np.zeros_like(dem)), np.float64), MIN_BUILDING_M)
    g = dem.shape[0]
    cell = float(layers.get("cell_m", 30.0))
    half = g * cell / 2.0
    kinds = list(KINDS)
    stack = np.stack([maps[k] for k in kinds]).ravel()
    p = stack / stack.sum()
    out = []
    for idx in rng.choice(p.size, size=int(n), replace=True, p=p):
        k, rc = divmod(int(idx), dem.size)
        r, c = divmod(rc, g)
        kind = kinds[k]
        if kind == "roofs":
            jx, jy = rng.uniform(-0.25, 0.25) * cell, rng.uniform(-0.2, 0.2) * cell
            z = dem[r, c] - med + bh[r, c]
        else:
            jx, jy = rng.uniform(-0.45, 0.45, 2) * cell
            z = None
        poses = POSES[kind]
        out.append({"r": r, "c": c, "kind": kind,
                    "x": float(-half + (c + 0.5) * cell + jx),
                    "y": float(half - (r + 0.5) * cell + jy),
                    "z": None if z is None else float(z),
                    "pose": poses[int(rng.integers(len(poses)))],
                    "yaw": float(rng.uniform(0, 2 * math.pi))})
    return out


# --------------------------------------------------------------- mission hook

def load_layers(path=DEFAULT_LAYERS):
    """Layer npz as a dict of arrays (meta_json left out)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "meta_json"}


def layer_cells(layers, grid):
    """Layer-grid (row, col, inside) for the centre of every mission cell.

    The layer npz stores its centre as `origin` = (lat, lon, alt) and its
    cell size as `cell_m`; the mission grid carries its own local frame.
    """
    g = grid.g
    rr, cc = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    e, n = grid.cell_center(rr, cc)
    lon, lat = grid.frame.to_lonlat(e, n)
    lat0, lon0 = float(layers["origin"][0]), float(layers["origin"][1])
    cell = float(layers["cell_m"])
    size = np.shape(layers["dem"])[0]
    dn = (np.asarray(lat) - lat0) * 111320.0
    de = (np.asarray(lon) - lon0) * 111320.0 * math.cos(math.radians(lat0))
    col = np.floor(de / cell + size / 2.0).astype(int)
    row = np.floor(size / 2.0 - dn / cell).astype(int)
    inside = (row >= 0) & (row < size) & (col >= 0) & (col < size)
    return np.clip(row, 0, size - 1), np.clip(col, 0, size - 1), inside


def to_mission_grid(maps, layers, grid, floor=UNIFORM_FLOOR):
    """Resample layer-grid prior components onto the mission grid (nearest cell).

    Components are densities, so nearest sampling keeps their shape for any
    mission cell size; mission cells outside the layer square get nothing
    but the uniform floor. Returns the same keys, blend summing to 1 in the mask.
    """
    row, col, inside = layer_cells(layers, grid)
    keep = inside & grid.mask
    comps = {k: _norm(np.where(keep, maps[k][row, col], 0.0)) for k in KINDS}
    shares = {k: float(maps[k].sum()) for k in KINDS}
    tot = sum(shares.values())
    shares = {k: v / tot for k, v in shares.items()} if tot > 0 else SHARES
    return _mix(comps, shares, grid.mask, floor)


def make_prior_builder(path=DEFAULT_LAYERS, fallback=None):
    """Orchestrator hook: `orch.prior_builder = make_prior_builder()`.

    Signature (sar/mission/orchestrator.py): (grid, lkp_en, lkp_time_h,
    scenario, water_level_m) -> {key: [g,g] map}. Flood missions get the
    flood prior from the layer npz at `path` (loaded once); any other
    scenario goes to `fallback` with the same signature, or the Koester ring
    prior when no fallback is given.
    """
    cache = {}

    def builder(grid, lkp_en, lkp_time_h, scenario, water_level_m):
        if scenario != "flood":
            if fallback is not None:
                return fallback(grid, lkp_en, lkp_time_h, scenario, water_level_m)
            from sar.mission.geo import koester_prior
            return {"koester": koester_prior(grid, lkp_en)}
        if "layers" not in cache:
            cache["layers"] = load_layers(path)
        lay = cache["layers"]
        maps = flood_prior(lay, float(water_level_m or 0.0))
        out = to_mission_grid(maps, lay, grid)
        return {k: out[k].astype(np.float32) for k in ("blend", *KINDS, "uniform")}

    return builder
