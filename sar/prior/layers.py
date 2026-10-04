"""Map layers for the real prior: cut, resample, cache and derive walker fields.

Plan section 6.1 lists the sources. Every region is stored as one compressed
npz with a metadata dict, keyed by its center in the local ENU frame.

Layer names inside an npz
-------------------------
dem          float32 [g,g]   ground elevation, m above the ellipsoid
worldcover   uint8   [g,g]   ESA WorldCover 10 m class (see `LC_*` below)
canopy       float32 [g,g]   canopy top height, m (0 where not forest)
trail        bool     [g,g]   path, footway, track, bridleway, steps, cycleway
road         bool     [g,g]   anything else in `highway`
stream       bool     [g,g]   stream or river centreline cells
water        bool     [g,g]   natural water plus wide streams and lakes
building     bool     [g,g]   any building footprint
lc           uint8   [g,g]   WorldCover class mapped to the small env index
slope        float32 [g,g]   gradient magnitude, unitless (rise over run)
walk         float32 [g,g]   Tobler walking speed on slope, m/s
blocked      bool     [g,g]   cells a walker cannot cross (water, cliffs)

Land cover index used by the env (`lc`)
---------------------------------------
The env needs a handful of classes, not the 11 WorldCover codes, so the codes
are folded down. `LC_NAMES` lists the index names and `LC_OF_CLASS` is the
lookup; `LC_NAMES[lc]` gives the name.

| Index | Name | ESA WorldCover v200 codes folded in |
|---|---|---|
| 0 | other | 20 shrubland |
| 1 | grass | 30 grassland, 40 cropland, 90 herbaceous wetland |
| 2 | forest | 10 tree cover, 95 mangroves |
| 3 | bare | 60 bare or sparse, 70 snow and ice, 100 moss and lichen |
| 4 | water | 80 permanent water |
| 5 | built | 50 built-up |

Trees are 10 and 95; the canopy layer falls back to these classes with a
default height where the ALS-GEDI product has no data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass

import numpy as np

CELL_M = 30.0

# ESA WorldCover 10 m class codes that map onto the small env index.
LC_OF_CLASS = {
    10: 2,   # tree cover
    20: 0,   # shrubland
    30: 1,   # grassland
    40: 1,   # cropland, open ground for walking and detection
    50: 5,   # built-up
    60: 3,   # bare or sparse vegetation
    70: 3,   # snow and ice
    80: 4,   # permanent water
    90: 1,   # herbaceous wetland
    95: 2,   # mangroves
    100: 3,  # moss and lichen
}
LC_NAMES = ["other", "grass", "forest", "bare", "water", "built"]
LC_WATER = 4
LC_FOREST = 2

# WorldCover tree classes get a default canopy height when the ALS-GEDI
# product is not used. See `CANOPY_SOURCE` in the metadata of each cut.
WORLDCOVER_TREE_CODES = (10, 95)
DEFAULT_TREE_HEIGHT_M = 20.0

# OSM tag sets. `out tags geom` keeps the response small.
TRAIL_VALUES = "^(path|footway|track|trail|steps|bridleway|cycleway)$"
ROAD_VALUES = (
    "^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
    "living_street|service|motorway_link|trunk_link|primary_link|secondary_link|"
    "tertiary_link)$"
)

DEM_BUCKET_URL = "https://s3.eu-central-1.amazonaws.com/copernicus-dem-30m"
WORLDCOVER_BUCKET_URL = "https://s3.eu-central-1.amazonaws.com/esa-worldcover/v200/2021/map"
CANOPY_BUCKET_URL = "https://s3.us-east-1.amazonaws.com/dataforgood-fb-data"
CANOPY_PREFIX = "forests/v1/alsgedi_global_v6_float_epsg4326_v3_10deg"
# 95th percentile height per 1/4000 degree cell: the canopy top, which the
# altitude floor needs. The cell average mixes in the ground between trees.
CANOPY_STAT = "p95"

OVERPASS_ENDPOINTS = (
    # mail.ru first: on 2026-10-23 it was the only mirror that answered a cut
    # query (17 s); overpass-api.de refused and the others timed out.
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
)


def tobler_speed(slope):
    """Tobler's hiking function: walking speed in m/s for a slope gradient.

    `slope` is rise over run and is negative downhill, positive uphill.
    """
    k = 6.0 * np.exp(-3.5 * np.abs(np.asarray(slope, np.float32) + 0.05))
    return (k * 1.34 / 3.6).astype(np.float32)  # km/h to m/s


def slope_from_dem(dem, cell_m=CELL_M):
    """Slope magnitude of the DEM as rise over run (always >= 0).

    Walkers move in all eight directions, so the walking speed and the cliff
    test use the steepest slope of the cell, not one axis of it.
    """
    g = dem.astype(np.float32)
    gy, gx = np.gradient(g, cell_m)
    return np.hypot(gx, gy).astype(np.float32)


def derive_fields(dem, worldcover, canopy, cell_m=CELL_M):
    """Fill the derived layers from the downloaded ones. Returns a dict."""
    lc = np.vectorize(lambda c: LC_OF_CLASS.get(int(c), 0), otypes=[np.uint8])(worldcover)
    slope = np.clip(slope_from_dem(dem, cell_m), 0.0, 2.0)
    walk = tobler_speed(slope).astype(np.float32)
    return {"lc": lc, "slope": slope, "walk": walk}


def refresh_derived(layers, cell_m=CELL_M):
    """Recompute slope, walk, blocked and speed_penalty from the raw layers.

    Cuts saved before a change to the derived fields stay usable: this runs on
    every load, so the walker always sees fields from the current code.
    """
    if "dem" not in layers or "lc" not in layers:
        return layers
    if "worldcover" in layers:
        layers["lc"] = np.vectorize(lambda c: LC_OF_CLASS.get(int(c), 0),
                                    otypes=[np.uint8])(layers["worldcover"])
    slope = np.clip(slope_from_dem(layers["dem"], cell_m), 0.0, 2.0)
    layers["slope"] = slope
    layers["walk"] = tobler_speed(slope)
    water = layers.get("water", np.zeros(slope.shape, bool))
    layers.update(blocking_fields(slope, water, layers["lc"]))
    return layers


def blocking_fields(slope, water, lc):
    """Walkability mask and a reduced speed over steep ground.

    Cliffs are anything above 45 degrees, which a hiker will not climb, and
    water cannot be walked through. A walker may still start or end next to
    water, so this only stops steps, not the start cell.
    """
    cliff = np.abs(slope) > 1.0
    blocked = (water | cliff | (lc == LC_WATER)).astype(bool)
    speed_penalty = np.where(np.abs(slope) > 0.7, 0.35, 1.0).astype(np.float32)
    return {"blocked": blocked, "speed_penalty": speed_penalty}


@dataclass(frozen=True)
class RegionMeta:
    """Everything needed to rebuild the npz for one region."""

    name: str
    lat: float
    lon: float
    cell_m: float
    size: int
    group: str = ""
    country: str = ""
    admin: str = ""
    ipp_lat: float = float("nan")
    ipp_lon: float = float("nan")
    ipp_rc: tuple = (0, 0)
    extras: dict = None

    def to_json(self):
        d = {
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "cell_m": self.cell_m,
            "size": self.size,
            "group": self.group,
            "country": self.country,
            "admin": self.admin,
            "ipp_lat": self.ipp_lat,
            "ipp_lon": self.ipp_lon,
            "ipp_rc": list(self.ipp_rc),
        }
        if self.extras:
            d.update(self.extras)
        return d

    @staticmethod
    def from_json(d):
        return RegionMeta(
            name=d["name"],
            lat=float(d["lat"]),
            lon=float(d["lon"]),
            cell_m=float(d["cell_m"]),
            size=int(d["size"]),
            group=d.get("group", ""),
            country=d.get("country", ""),
            admin=d.get("admin", ""),
            ipp_lat=float(d.get("ipp_lat", "nan")),
            ipp_lon=float(d.get("ipp_lon", "nan")),
            ipp_rc=tuple(d.get("ipp_rc", (0, 0))),
            extras=None,
        )


def tile_name(lat, lon, size=1):
    """Copernicus 1 degree DEM tile that contains a point.

    Tiles are named from their south west corner with a three digit
    longitude, for example `Copernicus_DSM_COG_10_N31_00_W111_00_DEM` covers
    latitude 31 to 32 and longitude -111 to -110.
    """
    lat0 = math.floor(lat)
    lon0 = math.floor(lon)
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    stem = f"Copernicus_DSM_COG_10_{ns}{abs(lat0):02d}_00_{ew}{abs(lon0):03d}_00_DEM"
    return stem, f"{stem}/{stem}.tif"


def dem_url(lat, lon):
    _, key = tile_name(lat, lon, 1)
    return f"{DEM_BUCKET_URL}/{key}"


def worldcover_url(lat, lon):
    """WorldCover 3 degree tile that contains a point.

    Tiles are named from their south west corner on a 3 degree grid, for
    example `N30W111` covers latitude 30 to 33 and longitude -111 to -108.
    """
    lat0 = 3 * math.floor(lat / 3.0)
    lon0 = 3 * math.floor(lon / 3.0)
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    name = f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}_Map.tif"
    return f"{WORLDCOVER_BUCKET_URL}/{name}"


def canopy_url(lat, lon, stat=CANOPY_STAT):
    """Meta/WRI ALS-GEDI 10 degree tile that contains a point.

    The bucket readme says tiles are named from their top left corner, so
    `lat=40.0_lon=-120.0` covers latitude 30 to 40 and longitude -120 to
    -110. Values are uint16 centimetres with 65535 as no data.
    """
    lat_top = 10.0 * math.floor(lat / 10.0) + 10.0
    lon0 = 10.0 * math.floor(lon / 10.0)
    key = f"{CANOPY_PREFIX}/meta_chm_lat={lat_top:.1f}_lon={lon0:.1f}_{stat}.tif"
    return f"{CANOPY_BUCKET_URL}/{key}"


def tile_points(bounds, step):
    """One point inside every `step` degree tile that a lon/lat box touches.

    The url builders take any point inside a tile, so this walks the tile
    grid over the box instead of guessing neighbours around the center.
    """
    w, s, e, n = bounds
    pts = []
    la = step * math.floor(s / step)
    while la < n:
        lo = step * math.floor(w / step)
        while lo < e:
            pts.append((la + step / 2.0, lo + step / 2.0))
            lo += step
        la += step
    return pts


def save_region(path, layers: dict, meta: RegionMeta, sources: dict):
    """Write one compressed npz plus the metadata json next to it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(layers)
    payload["meta_json"] = np.frombuffer(
        json.dumps({**meta.to_json(), "sources": sources}).encode(), dtype=np.uint8
    )
    np.savez_compressed(path, **payload)


def load_region(path):
    """Load a cut region. Returns (layers, meta)."""
    with np.load(path, allow_pickle=False) as z:
        layers = {k: z[k] for k in z.files if k != "meta_json"}
        meta = json.loads(bytes(z["meta_json"]).decode())
    refresh_derived(layers, float(meta.get("cell_m", CELL_M)))
    return layers, RegionMeta.from_json(meta)


def cache_key(lat, lon, size, cell_m=CELL_M):
    """Stable key for one cut, used for the download cache and for filenames."""
    q = f"{lat:.5f},{lon:.5f},{size},{cell_m:.0f}"
    return hashlib.sha1(q.encode()).hexdigest()[:16]


# Data values, not metres, that some of these rasters use.
# ALS-GEDI downsampled tiles are uint16 centimetres, 65535 is no data.
CANOPY_CM_TO_M = 0.01
CANOPY_NODATA = 65535


def read_cog_window(url, bounds, size, resampling="nearest", band=1, dst_dtype=None,
                    src_nodata=None):
    """Read a lon/lat square out of a remote COG, reprojecting if needed.

    Returns (array, meta). Handles a source that only covers part of the
    window by reading several tiles when the caller passes a list of urls.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    west, south, east, north = bounds
    out_dtype = dst_dtype or "float32"
    out = np.zeros((size, size), dtype=out_dtype)
    filled = np.zeros((size, size), dtype=bool)
    info = {"tiles": [], "covered": 0.0}
    for one in url if isinstance(url, (list, tuple)) else [url]:
        try:
            with rasterio.open("/vsicurl/" + one) as src:
                src_crs = "EPSG:4326" if src.crs.to_epsg() == 4326 else src.crs
                dtype = src.dtypes[band - 1]
                tile = np.zeros((size, size), dtype=dtype)
                reproject(
                    source=rasterio.band(src, band),
                    destination=tile,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    src_nodata=src.nodata if src_nodata is None else src_nodata,
                    dst_transform=rasterio.transform.from_bounds(
                        west, south, east, north, size, size
                    ),
                    dst_crs="EPSG:4326",
                    dst_nodata=0,
                    resampling=Resampling[resampling],
                )
                # rasterio fills everything when the window is only partly
                # covered, so only trust the cells inside the source footprint.
                win = rasterio.windows.from_bounds(west, south, east, north, src.transform)
                rows, cols = np.mgrid[0:size, 0:size]
                rr = win.row_off + (rows + 0.5) / size * win.height
                cc = win.col_off + (cols + 0.5) / size * win.width
                inside = (
                    (rr >= 0) & (rr < src.height) & (cc >= 0) & (cc < src.width)
                )
                new = inside & ~filled
                out[new] = tile[new]
                filled |= inside
                info["tiles"].append(one)
        except Exception:
            continue
    info["covered"] = float(filled.mean())
    return out.astype(out_dtype), filled, info

# ------------------------------------------------------------ env v2 terrain
# Env v2 (sar/config_v2.py LANDCOVER_NAMES) uses five Pd land cover classes:
# 0 water, 1 grass, 2 forest, 3 cropland, 4 built. They are taken straight from
# the WorldCover codes, so cropland keeps its own Pd class.
PD_OF_WORLDCOVER = {10: 2, 20: 1, 30: 1, 40: 3, 50: 4, 60: 1, 70: 1, 80: 0,
                    90: 1, 95: 2, 100: 1}
CANOPY_CLASS_EDGES_M = (8.0, 18.0)   # same thresholds as the synthetic terrain


def pd_landcover(worldcover, water=None, building=None):
    """WorldCover codes (plus OSM water and buildings) to the env Pd classes."""
    wc = np.asarray(worldcover)
    out = np.ones(wc.shape, np.int32)
    for code, cls in PD_OF_WORLDCOVER.items():
        out[wc == code] = cls
    if building is not None:
        out[np.asarray(building, bool)] = 4
    if water is not None:
        out[np.asarray(water, bool)] = 0
    return out


def canopy_class(height_m, building=None, bheight=None):
    """Canopy class exactly as `sar.env.v2.maps_v2.synthetic_terrain_np` sets it.

    Vegetation: 0 below 8 m, 1 from 8 to 18 m, 2 above 18 m. Building cells
    get 2, or 3 when the building is 18 m or taller, as the synthetic terrain
    gives buildings the dense or tall class.
    """
    h = np.asarray(height_m, np.float32)
    cls = np.digitize(h, CANOPY_CLASS_EDGES_M).astype(np.int32)
    if building is not None:
        b = np.asarray(building, bool)
        bh = np.zeros_like(h) if bheight is None else np.asarray(bheight, np.float32)
        cls = np.where(b, np.where(bh >= CANOPY_CLASS_EDGES_M[1], 3, 2), cls)
    return cls.astype(np.int32)


def env_terrain(layers, clearance_m=10.0):
    """The four per-cell arrays env v2 reads from a prior bank entry.

    `floor_m` is the altitude floor above local ground (obstacle top plus
    clearance), which is what the env compares with its level heights.
    `dem_m` is the ground height above the lowest cell of the cut, so the
    observation scalar `1 - dem/120` stays in range on real terrain.
    """
    dem = np.asarray(layers["dem"], np.float32)
    canopy = np.asarray(layers.get("canopy", np.zeros_like(dem)), np.float32)
    bheight = np.asarray(layers.get("bheight", np.zeros_like(dem)), np.float32)
    if "worldcover" in layers:
        lc = pd_landcover(layers["worldcover"], layers.get("water"), layers.get("building"))
    else:   # older cuts with only the six class index
        lc = np.choose(np.clip(layers["lc"], 0, 5), [1, 1, 2, 1, 0, 4]).astype(np.int32)
    obstacle = np.maximum(canopy, bheight)
    return {
        "landcover": lc.astype(np.int32),
        "canopy": canopy_class(canopy, layers.get("building"), bheight),
        "dem_m": (dem - float(dem.min())).astype(np.float32),
        "floor_m": (obstacle + clearance_m).astype(np.float32),
    }
