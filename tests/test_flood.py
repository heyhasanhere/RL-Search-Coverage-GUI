"""Flood prior and flood survivor sampler (plan 6.4, 13.3, M6).

A synthetic valley: a river along the west edge, a low town next to it, a
hill to the east with a few dry houses, and a road across. The real Lismore
layer file is checked too when it is present.
"""

import os

import numpy as np
import pytest

from sar.prior import flood as F

G = 32
CELL = 30.0


def make_layers():
    rr, cc = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    dem = (2.0 + 0.5 * cc).astype(np.float32)          # rises 0.5 m per cell going east
    river = cc < 2
    dem[river] = 0.0
    building = np.zeros((G, G), bool)
    building[8:24:2, 4:12:2] = True                     # low town, 2-6 m above the river
    building[8:24:4, 26:30:2] = True                    # hill houses, 15 m up
    road = np.zeros((G, G), bool)
    road[16, :] = True
    pop = np.where(building, 4.0, 0.0).astype(np.float32)
    return {"dem": dem, "river": river, "water": river, "building": building,
            "bheight": np.where(building, 6.0, 0.0).astype(np.float32), "road": road,
            "pop": pop, "cell_m": CELL, "origin": np.array([-28.81, 153.276, 0.0])}


def test_prior_sums_to_one_over_area():
    lay = make_layers()
    p = F.flood_prior(lay, 5.0)
    assert abs(p["blend"].sum() - 1.0) < 1e-9
    assert np.allclose(p["blend"], sum(p[k] for k in (*F.KINDS, "uniform")))
    area = np.zeros((G, G), bool)
    area[4:28, 2:30] = True
    pa = F.flood_prior(lay, 5.0, area=area)
    assert abs(pa["blend"].sum() - 1.0) < 1e-9
    assert pa["blend"][~area].sum() == 0.0
    shares = {k: pa[k].sum() for k in F.KINDS}
    assert shares["roofs"] > shares["high_ground"] > shares["water"] > 0


def test_flooded_roofs_get_mass_dry_roofs_do_not():
    lay = make_layers()
    flooded, depth = F.flood_extent(lay, 5.0)
    p = F.flood_prior(lay, 5.0)
    town = lay["building"] & flooded
    hill = lay["building"] & ~flooded
    assert town.sum() > 0 and hill.sum() > 0
    assert (p["roofs"][town] > 0).all()
    assert p["roofs"][hill].sum() == 0.0
    assert p["roofs"][~lay["building"]].sum() == 0.0


def test_submerged_building_has_no_roof_mass():
    lay = make_layers()
    flooded, depth = F.flood_extent(lay, 11.0)
    p = F.flood_prior(lay, 11.0)
    under = lay["building"] & flooded & (depth > 6.0 - F.ROOF_FREEBOARD_M)
    assert under.any()
    assert p["roofs"][under].sum() == 0.0


def test_dry_high_ground_gets_mass():
    lay = make_layers()
    flooded, _ = F.flood_extent(lay, 5.0)
    p = F.flood_prior(lay, 5.0)
    hg = p["high_ground"]
    assert hg[flooded].sum() == 0.0
    # the dry strip just east of the flood edge holds most of the high-ground mass
    edge = (~flooded) & (np.argmax(~flooded, axis=1)[:, None] + 3 >= np.arange(G)[None, :])
    assert hg[edge].sum() > 0.5 * hg.sum()
    assert hg[:, -3:].sum() < 0.05 * hg.sum()


def test_water_near_roads_only():
    lay = make_layers()
    flooded, _ = F.flood_extent(lay, 5.0)
    p = F.flood_prior(lay, 5.0)
    w = p["water"]
    assert w.sum() > 0
    assert w[lay["river"]].sum() == 0.0
    assert w[(np.abs(np.arange(G) - 16) > 1)[:, None] & np.ones((1, G), bool)].sum() == 0.0


def test_extent_needs_connection_to_river():
    lay = make_layers()
    lay["dem"][10:14, 20:24] = 0.5                       # a hollow behind higher ground
    flooded, _ = F.flood_extent(lay, 3.0)
    assert not flooded[10:14, 20:24].any()
    assert flooded[:, :2].all()


def test_water_level_moves_mass():
    lay = make_layers()
    lo, hi = F.flood_prior(lay, 5.0), F.flood_prior(lay, 8.0)
    flo, _ = F.flood_extent(lay, 5.0)
    fhi, _ = F.flood_extent(lay, 8.0)
    assert fhi.sum() > flo.sum() and (fhi | flo).sum() == fhi.sum()
    assert (hi["roofs"] > 0).sum() > (lo["roofs"] > 0).sum()
    cols = np.arange(G)
    centre = lambda m: float((m.sum(0) * cols).sum() / m.sum())   # noqa: E731
    assert centre(hi["high_ground"]) > centre(lo["high_ground"]) + 2.0
    assert np.abs(hi["blend"] - lo["blend"]).sum() > 0.5


def test_no_flood_is_not_nan():
    lay = make_layers()
    lay["road"][:] = False
    p = F.flood_prior(lay, 0.0)
    assert np.isfinite(p["blend"]).all() and abs(p["blend"].sum() - 1.0) < 1e-9


def test_sampler_differs_from_prior():
    lay = make_layers()
    p = F.flood_prior(lay, 5.0)
    s = F.sampler_map(lay, 5.0)
    ps = {k: p[k].sum() / (1 - p["uniform"].sum()) for k in F.KINDS}
    ss = {k: s[k].sum() for k in F.KINDS}
    assert abs(sum(ss.values()) - 1.0) < 1e-9
    # roofs against water and against high ground change by a factor of 2 x 2
    assert np.isclose((ss["roofs"] / ss["water"]) / (ps["roofs"] / ps["water"]), 0.25)
    assert np.isclose((ss["roofs"] / ss["high_ground"]) / (ps["roofs"] / ps["high_ground"]), 0.25)
    assert np.abs(s["blend"] - p["blend"]).sum() > 0.3

    rng = np.random.default_rng(0)
    vic = F.sample_flood_survivors(lay, 5.0, 2000, rng)
    got = {k: np.mean([v["kind"] == k for v in vic]) for k in F.KINDS}
    for k in F.KINDS:
        assert abs(got[k] - ss[k]) < 0.04
    flooded, depth = F.flood_extent(lay, 5.0)
    med = float(np.median(lay["dem"]))
    half = G * CELL / 2
    for v in vic[:200]:
        r, c = v["r"], v["c"]
        dx = abs(v["x"] - (-half + (c + 0.5) * CELL))
        dy = abs(v["y"] - (half - (r + 0.5) * CELL))
        if v["kind"] == "roofs":
            assert dx <= 0.25 * CELL + 1e-6 and dy <= 0.2 * CELL + 1e-6   # inside the Isaac box
            assert lay["building"][r, c] and flooded[r, c]
            assert np.isclose(v["z"], lay["dem"][r, c] - med + 6.0)
        elif v["kind"] == "water":
            assert flooded[r, c] and depth[r, c] <= F.SHALLOW_M and v["z"] is None
        else:
            assert not flooded[r, c]


def test_builder_maps_to_mission_grid():
    from sar.mission.geo import AreaGrid, Frame

    lay = make_layers()
    frame = Frame(lon0=153.276, lat0=-28.81)
    side = 0.004
    poly = [[153.276 - side, -28.81 - side], [153.276 + side, -28.81 - side],
            [153.276 + side, -28.81 + side], [153.276 - side, -28.81 + side]]
    grid = AreaGrid(frame, poly, CELL, 32)
    maps = F.flood_prior(lay, 5.0)
    out = F.to_mission_grid(maps, lay, grid)
    assert abs(out["blend"].sum() - 1.0) < 1e-6
    assert out["blend"][~grid.mask].sum() == 0.0
    # same frame, same cell size: roof cells land on the same cells
    row, col, inside = F.layer_cells(lay, grid)
    assert inside[grid.mask].all()
    assert (np.abs(row - np.arange(32)[:, None]) <= 1).all()
    assert out["roofs"].sum() > 0.4

    path = "/nonexistent.npz"
    b = F.make_prior_builder(path)
    wild = b(grid, None, 6.0, "wilderness", 0.0)
    assert set(wild) == {"koester"}


@pytest.mark.skipif(not os.path.exists(F.DEFAULT_LAYERS), reason="demo_flood.npz not built")
def test_demo_flood_file():
    lay = F.load_layers()
    for k in ("dem", "lc", "lc_names", "canopy", "building", "bheight", "water", "cell_m",
              "water_level", "origin", "pop", "river", "river_level", "stage_m"):
        assert k in lay, k
    stage = float(lay["stage_m"])
    p = F.flood_prior(lay, stage)
    assert abs(p["blend"].sum() - 1.0) < 1e-9
    assert p["roofs"].sum() > 0.3 and p["high_ground"].sum() > 0.1 and p["water"].sum() > 0.05
    assert np.allclose(p["blend"], lay["prior"], atol=1e-6)




def test_cells_builder_follows_survivor_mix():
    from types import SimpleNamespace
    from sar.prior.flood_bank import make_cells_builder
    # 2 x 2 grid of 10 m cells, one sample each: roof, dry, water, dry
    grid = SimpleNamespace(e0=0.0, n0=20.0, cell=10.0, g=2, mask=np.ones((2, 2), bool))
    cells = {"origin_en": [5.0, 15.0], "di_en": [10.0, 0.0], "dj_en": [0.0, -10.0], "nx": 2, "nz": 2, "types": "r.w."}

    def blend(shares):
        info = {"cells": cells, "survivor_shares": shares}
        return make_cells_builder(lambda: info)(grid, None, 0.0, "flood", 0.0)["blend"]
    roofs = {"roofs": 1.0, "high_ground": 0.0, "bridges": 0.0, "water": 0.0, "uniform": 0.0}
    assert np.allclose(blend(roofs), [[1, 0], [0, 0]])
    assert np.allclose(blend({**roofs, "roofs": 0.0, "water": 3.0}), [[0, 0], [1, 0]])
    # only kinds the scene lacks: anywhere, as the patch places them
    assert np.allclose(blend({**roofs, "roofs": 0.0, "bridges": 1.0}), 0.25)
    assert np.isclose(blend(None).sum(), 1.0)       # an old patch without survivor_shares: the default mix

