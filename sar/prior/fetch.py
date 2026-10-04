"""Cut map layers for a region from Copernicus DEM, WorldCover, ALS-GEDI and OSM.

Public COG tiles are read over HTTP with rasterio's virtual file system, so
only the window for the region is transferred. A cut near a tile edge is
assembled from the neighbouring tiles. OSM features come from the Overpass
API and are cached on disk by rounded bounding box.

`cut_region` is the entry point used by the scripts and tests.
"""

from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np

from . import layers as L

DEFAULT_CACHE = os.environ.get("SAR_PRIOR_CACHE", "runs/prior/cache")

# The ALS-GEDI product is a 10 degree global grid with gaps over Europe and
# North America. Where it has no data we fall back to the WorldCover tree
# class with a default height; `sources["canopy_source"]` records which was used.
CANOPY_SOURCE_ALS = "alsgedi_global_v6_float_10deg_p95"
CANOPY_SOURCE_FALLBACK = "worldcover_tree_class_default"
CANOPY_SOURCE_NOVEG = "none"

# GDAL options that make /vsicurl reads of a small window over a big COG much
# cheaper: no directory listing, no parallel range requests, a warm cache.
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIRANGE="NO",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    VSI_CACHE="TRUE",
    GDAL_CACHEMAX=512,
)


def gdal_env():
    """Rasterio environment used for every remote COG read."""
    from rasterio.env import Env

    return Env(**GDAL_ENV)


def _bounds(lat, lon, size, cell_m):
    """(west, south, east, north) of the square cut, from its center."""
    half = size * cell_m / 2.0
    deg_lat = half / 111320.0
    deg_lon = half / (111320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat


def _tile_urls(builder, bounds, step):
    """Urls of every tile on a `step` degree grid that the cut touches."""
    urls = []
    for la, lo in L.tile_points(bounds, step):
        u = builder(la, lo)
        if u not in urls:
            urls.append(u)
    return urls


def fetch_dem(lat, lon, size, cell_m, cache_dir):
    bounds = _bounds(lat, lon, size, cell_m)
    urls = _tile_urls(L.dem_url, bounds, 1.0)
    dem, filled, info = L.read_cog_window(urls, bounds, size, "bilinear",
                                          dst_dtype="float32")
    if info["covered"] < 0.999:
        raise RuntimeError(f"DEM covers only {info['covered']:.2f} of the cut "
                           f"(tiles read {len(info['tiles'])} of {len(urls)})")
    return dem, info["tiles"]


def fetch_worldcover(lat, lon, size, cell_m, cache_dir):
    bounds = _bounds(lat, lon, size, cell_m)
    urls = _tile_urls(L.worldcover_url, bounds, 3.0)
    wc, filled, info = L.read_cog_window(urls, bounds, size, "nearest",
                                         dst_dtype="uint8")
    if info["covered"] < 0.999:
        raise RuntimeError(f"WorldCover covers only {info['covered']:.2f} of the cut "
                           f"(tiles read {len(info['tiles'])} of {len(urls)})")
    return wc, info["tiles"]


def fetch_canopy(lat, lon, size, cell_m, cache_dir, worldcover=None):
    """ALS-GEDI canopy top height, or the WorldCover fallback where it has no data.

    Cells the product leaves as no data are filled from the WorldCover tree
    classes, so a partly covered cut still gets a usable layer.
    """
    bounds = _bounds(lat, lon, size, cell_m)
    urls = _tile_urls(L.canopy_url, bounds, 10.0)
    raw, filled, info = L.read_cog_window(
        urls, bounds, size, "average", dst_dtype="float32",
        src_nodata=L.CANOPY_NODATA,
    )
    valid = filled & (raw < L.CANOPY_NODATA)
    chm = np.where(valid, raw * L.CANOPY_CM_TO_M, 0.0).astype(np.float32)
    chm = np.clip(chm, 0.0, 60.0)
    tree = (np.isin(worldcover, L.WORLDCOVER_TREE_CODES) if worldcover is not None
            else np.zeros((size, size), bool))
    if valid.mean() > 0.5 and chm.max() > 0.5:
        hole = ~valid & tree
        chm = np.where(hole, L.DEFAULT_TREE_HEIGHT_M, chm).astype(np.float32)
        return chm, info["tiles"], CANOPY_SOURCE_ALS
    chm = np.where(tree, L.DEFAULT_TREE_HEIGHT_M, 0.0).astype(np.float32)
    return chm, info["tiles"], CANOPY_SOURCE_FALLBACK


def overpass_query(bounds):
    w, s, e, n = bounds
    pad = 0.02
    box = f"({s - pad},{w - pad},{n + pad},{e + pad})"
    return (
        f"[out:json][timeout:180];("
        f'way["highway"~"{L.TRAIL_VALUES}"]{box};'
        f'way["highway"~"{L.ROAD_VALUES}"]{box};'
        f'way["natural"="water"]{box};'
        f'way["waterway"~"^(stream|river)$"]{box};'
        f'way["building"]{box};'
        f'way["amenity"~"^(trailhead|wilderness_hut|alpine_hut)$"]{box};'
        f'node["amenity"~"^(trailhead|wilderness_hut|alpine_hut|drinking_water)$"]{box};'
        f");out tags geom;"
    )


def overpass_raw(query, cache_key=None, cache_dir=None, max_tries=6, sleep=15.0):
    """Run an arbitrary Overpass query, with the same retry and cache rules.

    `overpass_query` builds the fixed bbox query a cut needs. The bank builder
    needs a radius query around a seed instead, so this takes the query text.
    Returns the parsed JSON.
    """
    import requests

    path = None
    if cache_key and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"osmq_{cache_key}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

    headers = {"User-Agent": "sar-search-research/0.1"}
    last = ""
    for attempt in range(max_tries):
        for host in L.OVERPASS_ENDPOINTS:
            try:
                r = requests.post(host, data={"data": query}, headers=headers,
                                  timeout=240)
                if r.status_code == 200:
                    data = r.json()
                    if "elements" in data:
                        if path:
                            with open(path, "w") as f:
                                json.dump(data, f)
                        return data
                last = f"{host} {r.status_code}"
            except Exception as e:  # network hiccups and rate limits are expected
                last = f"{host} {type(e).__name__}"
        time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"Overpass failed after {max_tries} tries: {last}")


OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map"
_OVERPASS_DOWN = False       # set once every mirror refused the connection
OSM_API_SLEEP = 2.0          # seconds between calls, well inside the API policy
# Extra degrees fetched around a cut by the OSM API fallback, so a bank cut
# re-centred inside the probe square can reuse the probe's download.
OSM_API_PAD = 0.03
_WAY_AMENITY = ("trailhead", "wilderness_hut", "alpine_hut")
_NODE_AMENITY = _WAY_AMENITY + ("drinking_water",)


def _keep_way(tags):
    """Same tag rules as `overpass_query`, applied to raw OSM API ways."""
    hw = tags.get("highway", "")
    return bool(
        (hw and (re.match(L.TRAIL_VALUES, hw) or re.match(L.ROAD_VALUES, hw)))
        or tags.get("natural") == "water"
        or tags.get("waterway") in ("stream", "river")
        or "building" in tags
        or tags.get("amenity") in _WAY_AMENITY
    )


def parse_osm_xml(text):
    """OSM API 0.6 XML to the Overpass `out tags geom` JSON shape."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(text)
    nodes = {}
    elements = []
    for el in root.iter("node"):
        nid = el.get("id")
        lat, lon = float(el.get("lat")), float(el.get("lon"))
        nodes[nid] = (lat, lon)
        tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
        if tags.get("amenity") in _NODE_AMENITY:
            elements.append({"type": "node", "id": int(nid), "lat": lat,
                             "lon": lon, "tags": tags})
    for el in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
        if not _keep_way(tags):
            continue
        geom = [{"lat": nodes[n.get("ref")][0], "lon": nodes[n.get("ref")][1]}
                for n in el.findall("nd") if n.get("ref") in nodes]
        elements.append({"type": "way", "id": int(el.get("id")), "tags": tags,
                         "geometry": geom})
    return elements


def osm_box(lat, lon, size, cell_m):
    """The padded box the OSM API fallback downloads for a cut."""
    w, s, e, n = _bounds(lat, lon, size, cell_m)
    return (w - OSM_API_PAD, s - OSM_API_PAD, e + OSM_API_PAD, n + OSM_API_PAD)


def box_inside(inner, outer):
    """True when lon/lat box `inner` lies inside `outer`."""
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def osm_api_map(bounds, timeout=120, depth=0):
    """Features in a lon/lat box from the main OSM API `map` call.

    Fallback for when every Overpass mirror refuses or times out. The call
    is limited to 0.25 square degrees and 50 000 nodes, so a refused box is
    split in four and each quarter fetched on its own.
    """
    import requests

    w, s, e, n = bounds
    headers = {"User-Agent": "sar-search-research/0.1"}
    for attempt in range(6):
        r = requests.get(OSM_API_URL, params={"bbox": f"{w:.6f},{s:.6f},{e:.6f},{n:.6f}"},
                         headers=headers, timeout=timeout)
        time.sleep(OSM_API_SLEEP)
        if r.status_code not in (429, 509):
            break
        # Download quota: the API says how long to wait; back off further.
        time.sleep(15.0 * (attempt + 1))
    if r.status_code == 400 and depth < 3:
        mx, my = (w + e) / 2.0, (s + n) / 2.0
        out = {}
        for q in ((w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)):
            for el in osm_api_map(q, timeout, depth + 1):
                out[(el["type"], el["id"])] = el
        return list(out.values())
    if r.status_code != 200:
        raise RuntimeError(f"OSM API {r.status_code}: {r.text[:120]}")
    return parse_osm_xml(r.text)


def _overpass_post(query, timeout, connect_timeout=10):
    """One pass over the Overpass mirrors. Returns (data or None, reason)."""
    import requests

    headers = {"User-Agent": "sar-search-research/0.1"}
    last = ""
    for host in L.OVERPASS_ENDPOINTS:
        try:
            r = requests.post(host, data={"data": query}, headers=headers,
                              timeout=(connect_timeout, timeout))
            if r.status_code == 200:
                data = r.json()
                if "elements" in data:
                    return data, "ok"
            last = f"{host} HTTP {r.status_code}"
        except Exception as e:  # refused, timed out or rate limited
            last = f"{host} {type(e).__name__}"
    return None, last


def fetch_osm(lat, lon, size, cell_m, cache_dir, max_tries=6, sleep=20.0,
              timeout=240):
    """OSM features for the square, cached on disk by rounded bbox.

    Overpass is tried first with a short connect timeout. If every mirror
    refuses (seen on 2026-10-23: `overpass-api.de` refused port 443 from both
    machines and the other mirrors timed out), the main OSM API `map` call
    is used instead with the same tag rules. `SAR_OSM_SOURCE=osm_api` skips
    Overpass entirely. Raises RuntimeError with the reason if both fail.
    """
    bounds = _bounds(lat, lon, size, cell_m)
    key = L.cache_key(round(bounds[1], 3), round(bounds[0], 3), size, cell_m)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"osm_{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f), path

    global _OVERPASS_DOWN
    reasons = []
    source = os.environ.get("SAR_OSM_SOURCE", "auto")
    if source != "osm_api" and not _OVERPASS_DOWN:
        q = overpass_query(bounds)
        for attempt in range(max_tries):
            data, why = _overpass_post(q, timeout)
            if data is not None:
                data["source"] = "overpass"
                with open(path, "w") as f:
                    json.dump(data, f)
                return data, path
            reasons.append(why)
            if "ConnectionError" in why or "ConnectTimeout" in why:
                # A refused or unreachable mirror does not recover within a
                # retry; remember it for the rest of this process.
                _OVERPASS_DOWN = True
                break
            if attempt < max_tries - 1:
                time.sleep(sleep * (attempt + 1))
    w, s, e, n = bounds
    pad = OSM_API_PAD
    box = (w - pad, s - pad, e + pad, n + pad)
    for attempt in range(3):
        try:
            data = {"elements": osm_api_map(box, timeout=timeout), "source": "osm_api"}
            with open(path, "w") as f:
                json.dump(data, f)
            return data, path
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"osm_api {type(exc).__name__}: {str(exc)[:80]}")
            time.sleep(sleep * (attempt + 1))
    raise RuntimeError("OSM fetch failed: " + " | ".join(reasons[-4:]))


def _linestring(ring):
    return {"type": "LineString", "coordinates": [list(p) for p in ring]}


def _polygon(ring):
    coords = [list(p) for p in ring]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return {"type": "Polygon", "coordinates": [coords]}


def rasterize_osm(data, bounds, size, cell_m):
    """Burn OSM ways and points onto the size x size grid."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    w, s, e, n = bounds
    shape = (size, size)
    transform = from_bounds(w, s, e, n, size, size)
    layers = {k: [] for k in ("trail", "road", "stream", "river", "water", "building")}
    pois = []
    bh_items = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "node" and "lat" in el:
            lon, lat = el["lon"], el["lat"]
            if w <= lon <= e and s <= lat <= n:
                pois.append({"amenity": tags.get("amenity", ""), "lon": lon, "lat": lat})
            continue
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        ring = [(p["lon"], p["lat"]) for p in geom]
        hw = tags.get("highway")
        if hw and re.match(L.TRAIL_VALUES, hw):
            layers["trail"].append((_linestring(ring), 1))
        elif hw:
            layers["road"].append((_linestring(ring), 1))
        elif tags.get("natural") == "water":
            layers["water"].append((_polygon(ring), 1))
        elif tags.get("waterway") in ("stream", "river"):
            layers["stream"].append((_linestring(ring), 1))
            if tags.get("waterway") == "river":
                layers["river"].append((_linestring(ring), 1))
        if "building" in tags:
            layers["building"].append((_polygon(ring), 1))
            bh_items.append((_polygon(ring), building_height(tags)))

    out = {}
    for name, items in layers.items():
        if not items:
            out[name] = np.zeros(shape, bool)
            continue
        arr = rasterize(
            items, out_shape=shape, transform=transform, fill=0,
            dtype="uint8", all_touched=False,
        )
        out[name] = arr.astype(bool)
    if bh_items:
        # Later shapes overwrite earlier ones; sort so the tallest wins a cell.
        bh_items.sort(key=lambda it: it[1])
        out["bheight"] = rasterize(
            bh_items, out_shape=shape, transform=transform, fill=0.0,
            dtype="float32", all_touched=False,
        ).astype(np.float32)
    else:
        out["bheight"] = np.zeros(shape, np.float32)
    # A river centreline blocks walking; a stream or desert wash does not.
    # (Before 2026-10-23 every stream cell was burnt in as water, which in the
    # Arizona cases blocked about 7 percent of the cut along dry washes.)
    out["water"] = out["water"] | out.pop("river")
    out["pois"] = pois
    return out


BUILDING_LEVEL_M = 3.0
BUILDING_DEFAULT_LEVELS = 2       # plan 6.1: default 2 levels when OSM has no height


def building_height(tags):
    """Building height in metres from OSM `height` or `building:levels`."""
    h = tags.get("height", "")
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", h)
    if m:
        return float(m.group(1))
    lv = tags.get("building:levels", "")
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", lv)
    if m:
        return float(m.group(1)) * BUILDING_LEVEL_M
    return BUILDING_DEFAULT_LEVELS * BUILDING_LEVEL_M


def _wide(mask, radius):
    """Binary dilation of a mask by `radius` cells (cheap max filter)."""
    import scipy.ndimage as ndi

    return ndi.binary_dilation(mask, iterations=radius)


def cut_region(lat, lon, size=192, cell_m=L.CELL_M, cache_dir=DEFAULT_CACHE,
               with_osm=True, name="region", ipp_rc=None, extra_meta=None,
               osm_timeout=240, osm_tries=6, osm=None):
    """Cut every layer for one square region and return (layers, meta, extra).

    `osm` is an already fetched feature set that covers the cut box; when it
    is given no new OSM request is made.
    """
    tile_cache = os.path.join(cache_dir, "tiles")
    with gdal_env():
        dem, dem_tiles = fetch_dem(lat, lon, size, cell_m, tile_cache)
        wc, wc_tiles = fetch_worldcover(lat, lon, size, cell_m, tile_cache)
        canopy, chm_tiles, chm_source = fetch_canopy(lat, lon, size, cell_m, tile_cache, wc)

    out = {"dem": dem, "worldcover": wc, "canopy": canopy,
           "lc_names": np.array(L.LC_NAMES)}
    out.update(L.derive_fields(dem, wc, canopy, cell_m))

    pois = []
    if with_osm:
        if osm is None:
            osm, _ = fetch_osm(lat, lon, size, cell_m, tile_cache,
                               max_tries=osm_tries, timeout=osm_timeout)
        burn = rasterize_osm(osm, _bounds(lat, lon, size, cell_m), size, cell_m)
        pois = burn.pop("pois")
        out.update(burn)
    else:
        z = np.zeros((size, size), bool)
        out.update({"trail": z, "road": z.copy(), "stream": z.copy(),
                    "water": z.copy(), "building": z.copy(),
                    "bheight": np.zeros((size, size), np.float32)})

    out.update(L.blocking_fields(out["slope"], out["water"], out["lc"]))
    out["floor"] = (dem + np.maximum(canopy, out["bheight"]) + 10.0).astype(np.float32)

    meta = L.RegionMeta(
        name=name, lat=lat, lon=lon, cell_m=cell_m, size=size,
        ipp_rc=tuple(ipp_rc) if ipp_rc is not None else (size // 2, size // 2),
        ipp_lat=lat, ipp_lon=lon, extras=extra_meta,
    )
    sources = {
        "dem": dem_tiles, "worldcover": wc_tiles, "canopy": chm_tiles,
        "canopy_source": chm_source,
    }
    return out, meta, {"sources": sources, "pois": pois}