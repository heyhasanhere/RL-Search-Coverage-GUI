"""Synthetic flood towns and the cell-type flood prior.

A flood prior here is a mix of five components over 30 m cells:

| Key | Who | Cells |
|---|---|---|
| `roofs` | People on the roofs of flooded buildings | Flooded buildings whose roof is above water |
| `high_ground` | People who left flooded buildings | Dry cells near the flood, islands doubled |
| `bridges` | People on bridges | Road cells over the normal river |
| `water` | People caught in the water | Flooded cells off the river, next to a road first |
| `uniform` | Anyone anywhere | Every cell in the area |

Nobody is inside a building. `cell_prior` builds the prior from per-cell
type maps, so the same function serves a synthetic town and a simulator scene
that exports its cell types (`make_cells_builder`, the FloodUAV scene). `random_town` draws a town around a river, and
`make_bank` writes many of them as a training bank (`prior`, `ipp_rc`) for
`sar.train.train --real-bank`.

    uv run python -m sar.prior.flood_bank --out data/regions/flood_train.npz --n 600 --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np
import scipy.ndimage as ndi

from sar.prior.flood import ISLAND_BONUS, HIGH_REACH_CELLS, HIGH_SIGMA_CELLS, ROOF_FREEBOARD_M, flood_extent

SHARES = {"roofs": 0.50, "high_ground": 0.25, "bridges": 0.10, "water": 0.10, "uniform": 0.05}
KINDS = ("roofs", "high_ground", "bridges", "water")
WATER_ROAD_WEIGHT = 3.0         # a flooded cell next to a road holds this many times more people


def _norm(a):
    s = float(a.sum())
    return a / s if s > 0 else np.zeros_like(a, np.float64)


def _islands(dry):
    lab, n = ndi.label(dry, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.zeros_like(dry)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return (lab > 0) & ~np.isin(lab, edge)


def cell_components(roof, flooded_bld, flooded, bridge, road, river):
    """Unit-mass map per component from cell-type maps (bool, or 0..1 cover for roof and bridge)."""
    roof_w, bridge_w = np.asarray(roof, np.float64), np.asarray(bridge, np.float64)
    dry = ~flooded
    spread = ndi.gaussian_filter(np.asarray(flooded_bld, np.float64), HIGH_SIGMA_CELLS, mode="constant")
    near = ndi.distance_transform_edt(dry) <= HIGH_REACH_CELLS
    high = np.where(dry & near, spread, 0.0) * np.where(_islands(dry), ISLAND_BONUS, 1.0)
    wet = flooded & ~river & ~(roof_w > 0) & ~(bridge_w > 0)
    by_road = ndi.binary_dilation(road, iterations=1)
    water = wet * np.where(by_road, WATER_ROAD_WEIGHT, 1.0)
    return {"roofs": _norm(roof_w), "high_ground": _norm(high),
            "bridges": _norm(bridge_w), "water": _norm(water)}


def cell_prior(comps, area, shares=SHARES):
    """Blend unit maps; an empty component gives its share to the others. Sums to 1 over area."""
    area = np.asarray(area, bool)
    live = {k: shares[k] for k in KINDS if (comps[k] * area).sum() > 0}
    tot = sum(live.values()) + shares["uniform"]
    if tot <= 0:                    # a mix of only kinds the scene lacks: anywhere, as Layout.cs places them
        return _norm(area.astype(np.float64))
    out = shares["uniform"] / tot * _norm(area.astype(np.float64))
    for k, s in live.items():
        out = out + s / tot * _norm(comps[k] * area)
    return out


def cells_on_grid(cells, grid):
    """Cover fraction per mission cell of each sample type (Layout.cs codes) -> {code: [g,g]}, count [g,g]."""
    i, j = np.meshgrid(np.arange(cells["nx"]), np.arange(cells["nz"]), indexing="xy")
    o, di, dj = (np.asarray(cells[k], float) for k in ("origin_en", "di_en", "dj_en"))
    e = o[0] + i * di[0] + j * dj[0]
    n = o[1] + i * di[1] + j * dj[1]
    c = np.floor((e.ravel() - grid.e0) / grid.cell).astype(int)
    r = np.floor((grid.n0 - n.ravel()) / grid.cell).astype(int)
    ok = (r >= 0) & (r < grid.g) & (c >= 0) & (c < grid.g)
    t = np.frombuffer(cells["types"].encode(), np.uint8)
    g = grid.g
    count = np.zeros((g, g))
    np.add.at(count, (r[ok], c[ok]), 1.0)
    out = {}
    for code in ".wrsbho":
        m = np.zeros((g, g))
        np.add.at(m, (r[ok], c[ok]), (t[ok] == ord(code)).astype(float))
        out[code] = m / np.maximum(count, 1.0)
    return out, count


def cells_prior(cells, grid, shares=SHARES):
    """Flood prior components on the mission grid from scene cell types."""
    f, count = cells_on_grid(cells, grid)
    roof, bridge = f["r"], f["b"]
    fb = (f["r"] + f["s"]) > 0
    flooded = (f["w"] + f["r"] + f["s"] + f["b"]) >= 0.5
    zero = np.zeros_like(flooded)
    comps = cell_components(roof, fb, flooded, bridge, zero, zero)
    area = np.asarray(grid.mask, bool) & (count > 0)
    blend = cell_prior(comps, area, shares)
    live = sum(shares[k] for k in KINDS if (comps[k] * area).sum() > 0) + shares["uniform"]
    if live <= 0:                   # see cell_prior
        shares, live = {**{k: 0.0 for k in KINDS}, "uniform": 1.0}, 1.0
    out = {k: shares[k] / live * _norm(comps[k] * area) for k in KINDS}
    out["uniform"] = shares["uniform"] / live * _norm(area.astype(np.float64))
    out["blend"] = blend
    return {k: v.astype(np.float32) for k, v in out.items()}


def make_cells_builder(get_info):
    """Orchestrator prior builder from the FloodUAV scene cell types; `get_info()` gives the bridge /info.

    The shares are the survivor terrain mix the scene's survivors were placed with (`survivor_shares`),
    else SHARES. Without cell types (an old patch) it falls back to the Koester ring prior.
    """
    def builder(grid, lkp_en, lkp_time_h, scenario, water_level_m):
        info = get_info() or {}
        cells = info.get("cells")
        if not cells:
            from sar.mission.geo import koester_prior
            return {"koester": koester_prior(grid, lkp_en)}
        mix = info.get("survivor_shares")
        if isinstance(mix, dict) and set(mix) == set(SHARES) and sum(mix.values()) > 0:
            shares = {k: float(mix[k]) / sum(mix.values()) for k in SHARES}
        else:
            shares = SHARES
        return cells_prior(cells, grid, shares)
    return builder


# --------------------------------------------------------------- synthetic towns

def _river(rng, n):
    """Meandering river mask and its centreline distance map."""
    th = rng.uniform(0, np.pi)
    d, u = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    c = n / 2 + rng.uniform(-8, 8, 2)
    t = np.linspace(-n, n, 4 * n)
    off = sum(rng.uniform(2, 14) * np.sin(2 * np.pi * t / rng.uniform(40, 160) + rng.uniform(0, 6.3))
              for _ in range(3))
    pts = c + t[:, None] * d + off[:, None] * u
    line = np.zeros((n, n), bool)
    ij = np.round(pts).astype(int)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    line[ij[ok, 0], ij[ok, 1]] = True
    dist = ndi.distance_transform_edt(~line)
    width = rng.uniform(0.6, 2.2)
    return dist <= width, dist


def _blob(rng, n, rc, radius, density):
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - rc[0], xx - rc[1])
    p = density * np.exp(-(r / radius) ** 2)
    return rng.random((n, n)) < p


def random_town(rng, n=192, cell_m=30.0):
    """One synthetic flood town. Returns (layers dict, water level m, ipp (r, c))."""
    river, rdist = _river(rng, n)
    noise = ndi.gaussian_filter(rng.normal(size=(n, n)), rng.uniform(3, 8))
    noise *= rng.uniform(0.5, 3.0) / max(noise.std(), 1e-9)
    dem = rdist * cell_m * rng.uniform(0.002, 0.012) + noise
    for _ in range(rng.integers(2, 9)):                 # hills, some become islands
        rc = rng.uniform(0, n, 2)
        yy, xx = np.mgrid[:n, :n]
        dem += rng.uniform(2, 9) * np.exp(-((yy - rc[0]) ** 2 + (xx - rc[1]) ** 2) / (2 * rng.uniform(2, 8) ** 2))

    # Town centre close to the river, near the map middle.
    cand = np.argwhere((rdist > 1.5) & (rdist < rng.uniform(4, 12)))
    mid = np.abs(cand - n / 2).max(1) < 20
    ipp = cand[rng.choice(np.flatnonzero(mid))] if mid.any() else np.array([n // 2, n // 2])
    rad = rng.uniform(6, 16)
    bld = _blob(rng, n, ipp, rad, rng.uniform(0.4, 0.8))
    for _ in range(rng.integers(1, 6)):                 # villages
        rc = ipp + rng.normal(0, 30, 2)
        bld |= _blob(rng, n, rc, rng.uniform(2, 5), rng.uniform(0.3, 0.7))

    # Roads: a street grid in town plus long roads through it that cross the river.
    road = np.zeros((n, n), bool)
    yy, xx = np.mgrid[:n, :n]
    step = rng.integers(3, 6)
    in_town = np.hypot(yy - ipp[0], xx - ipp[1]) < rad * 1.3
    road |= in_town & (((yy - ipp[0]) % step == 0) | ((xx - ipp[1]) % step == 0))
    for _ in range(rng.integers(1, 4)):
        th = rng.uniform(0, np.pi)
        o = ipp + rng.normal(0, rad / 2, 2)
        road |= np.abs((yy - o[0]) * np.cos(th) - (xx - o[1]) * np.sin(th)) < 0.5
    bld &= ~road & ~river

    bheight = np.where(bld, rng.choice([3.0, 4.0, 6.0, 7.0, 10.0, 12.0], (n, n),
                                       p=[0.25, 0.3, 0.2, 0.12, 0.08, 0.05]), 0.0)
    layers = {"dem": dem, "river": river, "water": river, "building": bld, "bheight": bheight,
              "road": road, "cell_m": cell_m}
    return layers, float(rng.uniform(1.0, 5.0)), (int(ipp[0]), int(ipp[1]))


def town_prior(layers, water_level_m, shares=SHARES):
    """Cell types of a town at a water level, then the prior over the whole map."""
    flooded, depth = flood_extent(layers, water_level_m)
    river = np.asarray(layers["river"], bool)
    road = np.asarray(layers["road"], bool)
    bld = np.asarray(layers["building"], bool)
    fb = flooded & bld & ~river
    roof = fb & (layers["bheight"] > depth + ROOF_FREEBOARD_M)
    bridge = road & river
    comps = cell_components(roof, fb, flooded, bridge, road, river)
    return cell_prior(comps, np.ones_like(river), shares), comps


def jitter(rng, shares=SHARES, conc=30.0):
    """Shares drawn around the target so the agent does not rely on exact values."""
    keys = list(shares)
    w = rng.dirichlet(conc * np.array([shares[k] for k in keys]))
    return dict(zip(keys, w))


def make_bank(n_maps, seed, size=192, jitter_shares=True):
    rng = np.random.default_rng(seed)
    pri, ipps, stats = [], [], []
    while len(pri) < n_maps:
        layers, wl, ipp = random_town(rng, size)
        sh = jitter(rng) if jitter_shares else SHARES
        p, comps = town_prior(layers, wl, sh)
        if comps["roofs"].sum() == 0:                   # a town that did not flood is no flood case
            continue
        pri.append(p.astype(np.float32))
        ipps.append(ipp)
        stats.append([float(comps[k].sum() > 0) for k in KINDS])
    return np.stack(pri), np.array(ipps, np.int32), np.array(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--no-jitter", action="store_true")
    a = ap.parse_args()
    pri, ipp, st = make_bank(a.n, a.seed, a.size, not a.no_jitter)
    np.savez_compressed(a.out, prior=pri, ipp_rc=ipp)
    print(f"{a.out}: {len(pri)} maps, components present: "
          + ", ".join(f"{k} {v:.0%}" for k, v in zip(KINDS, st.mean(0))))


if __name__ == "__main__":
    main()
