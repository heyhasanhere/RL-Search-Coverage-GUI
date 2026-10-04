"""Prior tests from plan section 19.

Covers the four checks the plan names for the prior:

- walker end points never inside water bodies
- the blend sums to the expected survivor count
- leave-one-out never uses the held-out case
- a mixture fitted on cached pure-strategy maps equals a direct walker run
  with that mixture (statistical check)

Everything runs on a small synthetic terrain, so the tests do not need the
downloaded cases or a GPU.
"""

import numpy as np
import pytest

from sar.prior import fit as F
from sar.prior import koester as K
from sar.prior import layers as L
from sar.prior import service
from sar.prior import walker as W

G = 32
TIMES = (1.0, 3.0, 6.0)


def make_layers(seed=0, water=True, trails=True):
    """A synthetic region with a lake, some trails and gentle hills."""
    rng = np.random.default_rng(seed)
    rr, cc = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    dem = (60.0 * np.sin(rr / 5.0) + 40.0 * np.cos(cc / 7.0)
           + rng.normal(0, 3, (G, G))).astype(np.float32)
    wc = np.full((G, G), 30, np.uint8)        # grassland
    wc[dem > np.median(dem)] = 10             # tree cover on the high ground
    canopy = np.where(wc == 10, 22.0, 0.0).astype(np.float32)
    lay = {"dem": dem, "worldcover": wc, "canopy": canopy}
    lay.update(L.derive_fields(dem, wc, canopy, L.CELL_M))

    wet = np.zeros((G, G), bool)
    if water:
        wet[G // 2 - 2:G // 2 + 2, G // 3:G // 3 + 6] = True
    trail = np.zeros((G, G), bool)
    if trails:
        trail[:, G // 2] = True
        trail[G // 2, :] = True
    lay["water"] = wet
    lay["trail"] = trail
    lay["road"] = np.zeros((G, G), bool)
    lay["stream"] = np.zeros((G, G), bool)
    lay["building"] = np.zeros((G, G), bool)
    lay.update(L.blocking_fields(lay["slope"], wet, lay["lc"]))
    lay["floor"] = (dem + canopy + 10.0).astype(np.float32)
    return lay


def ipp_rc(lay):
    """A free start cell near the centre (border cells are dropped as exits)."""
    idx = np.argwhere(~lay["blocked"])
    d = np.abs(idx - (G // 2 - 3)).sum(1)
    return tuple(int(x) for x in idx[np.argmin(d)])


def find_rc(lay, rng):
    """A random free cell to stand in for a real find location."""
    free = np.argwhere(~lay["blocked"])
    return tuple(int(x) for x in free[rng.integers(len(free))])


def test_walker_endpoints_never_in_water():
    """Plan 19: walker end points never inside water bodies."""
    lay = make_layers(water=True)
    terrain = W.Terrain(lay)
    start = ipp_rc(lay)
    hist = W.run_pure_strategies(
        terrain, start, n_per_strategy=200, times_h=TIMES, seed=3
    )
    assert hist.shape == (len(W.STRATEGIES), len(TIMES), G, G)
    assert hist.sum() > 0
    water = lay["water"] | (lay["lc"] == 4)
    assert hist[:, :, water].sum() == 0, "a walker ended inside water"
    # Blocked cells are cliffs as well as water, and are closed too.
    assert hist[:, :, lay["blocked"]].sum() == 0


def test_walker_stay_put_stays_at_the_start():
    """The stay-put strategy has all its end points on the start cell."""
    lay = make_layers(water=False)
    terrain = W.Terrain(lay)
    start = ipp_rc(lay)
    hist = W.run_pure_strategies(terrain, start, n_per_strategy=50, times_h=TIMES)
    sp = hist[W.STRATEGY_ID["SP"]]
    assert sp.sum() == 50 * len(TIMES)
    assert sp[:, start[0], start[1]].sum() == sp.sum()


def test_blend_sums_to_expected_count():
    """Plan 19: the blend sums to the expected survivor count."""
    lay = make_layers(seed=1)
    start = ipp_rc(lay)
    params = {
        "trail_pref": 1.0, "stop_prob": 0.0, "hours": 3.0, "times_h": list(TIMES),
        "weights": {"uniform": 0.05, "walker": 0.5, "koester": 0.45},
        "strategy_mix": [0.1, 0.4, 0.5, 0.0, 0.0, 0.0],
    }
    cached = W.run_pure_strategies(
        W.Terrain(lay), start, n_per_strategy=200, times_h=TIMES
    )
    for count in (1.0, 2.0, 3.0):
        prior = service.build_prior(
            lay, start, 6.0, params, expected_count=count,
            times_h=TIMES, walker_cache=cached,
        )
        assert prior.dtype == np.float32
        assert prior.shape == (G, G)
        assert prior[lay["blocked"]].sum() == 0.0
        assert prior.sum() == pytest.approx(count, rel=1e-4)


def test_koester_sums_to_expected_count():
    """The baseline component alone also normalizes."""
    lay = make_layers(seed=2)
    start = ipp_rc(lay)
    m = K.koester_map(lay, start, expected_count=1.5)
    assert m.sum() == pytest.approx(1.5, rel=1e-4)
    assert m[lay["blocked"]].sum() == 0.0


def test_distance_share_is_within_the_area():
    """The area share a search must cover is a fraction of the grid."""
    lay = make_layers(seed=3)
    m = K.koester_map(lay, ipp_rc(lay))
    share = K.distance_share(m, (5, 5))
    assert 0.0 < share <= 1.0


def _synthetic_values(n=12, seed=0):
    """Find-cell values for n cases drawn from a known blend."""
    rng = np.random.default_rng(seed)
    n_grid, n_t, n_comp = 2, 2, 8
    vals = rng.uniform(0.0, 2e-3, (n, n_grid, n_t, n_comp))
    vals[..., 0] = 1.0 / (G * G)
    return vals


def test_em_weights_are_on_the_simplex_with_the_floor():
    vals = _synthetic_values()[:, 0, 0]
    for allowed in F.VARIANTS.values():
        w = F.em_weights(vals, allowed)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert w[0] == pytest.approx(F.UNIFORM_FLOOR)
        assert (w >= 0).all()
        assert np.all(w[1:][~np.asarray(allowed)[1:]] == 0.0)


def test_em_weights_recover_a_known_mixture():
    """Cases drawn from components with known weights give those weights back."""
    rng = np.random.default_rng(3)
    g2 = 400
    comps = rng.dirichlet(np.ones(g2) * 0.3, size=3)        # three component maps
    true = np.array([0.6, 0.3, 0.1])
    prior = 0.05 / g2 + 0.95 * true @ comps
    cells = rng.choice(g2, size=4000, p=prior / prior.sum())
    vals = np.zeros((len(cells), 8))
    vals[:, 0] = 1.0 / g2
    vals[:, 1:4] = comps[:, cells].T
    w = F.em_weights(vals, [False, True, True, True, False, False, False, False])
    assert np.allclose(w[1:4] / 0.95, true, atol=0.05)


def test_loo_never_uses_the_held_out_case(monkeypatch):
    """Plan 19: leave-one-out never uses the held-out case.

    Every fit call is recorded; the training rows of the fold that scores case
    i must never contain i. Changing the held-out case's own values must also
    leave its fitted weights unchanged.
    """
    vals = _synthetic_values(n=8)
    seen = []
    real_fit = F.fit_on

    def spy(v, train_idx, variant):
        seen.append(set(int(j) for j in train_idx))
        return real_fit(v, train_idx, variant)

    monkeypatch.setattr(F, "fit_on", spy)
    maps = np.random.default_rng(1).uniform(0, 1, (8, G, G))

    def maps_fn(i, gi, ti):
        return maps
    report, rows = F.run_loo(vals, maps_fn, [(3, 3)] * 8)
    n_var = len(F.VARIANTS)
    assert len(seen) == 8 * n_var
    for i in range(8):
        for k in range(n_var):
            assert i not in seen[i * n_var + k]
            assert len(seen[i * n_var + k]) == 7
    for name in F.VARIANTS:
        assert report[name]["n"] == 8
        assert 0.0 <= report[name]["top10"] <= 1.0

    monkeypatch.setattr(F, "fit_on", real_fit)
    v2 = vals.copy()
    v2[0] *= 50.0                                          # wild change to case 0 only
    _, rows_a = F.run_loo(vals, maps_fn, [(3, 3)] * 8)
    _, rows_b = F.run_loo(v2, maps_fn, [(3, 3)] * 8)
    assert rows_a["blend"][0]["weights"] == rows_b["blend"][0]["weights"]


def test_score_share_and_loglik():
    p = np.zeros((G, G))
    p[0, 0] = 0.5
    p[1, 1] = 0.25
    p += 0.25 / (G * G)
    share, ll = F.score(p, (0, 0))
    assert share == pytest.approx(1.0 / (G * G))
    assert ll == pytest.approx(np.log(0.5 + 0.25 / (G * G)))
    share_low, _ = F.score(p, (5, 5))
    assert share_low == pytest.approx(1.0)


def test_mixture_of_cached_equals_direct_run():
    """Plan 19: a mixture fitted on cached maps equals a direct run.

    The cached pure-strategy maps are sampled counts, so mixing them by the
    strategy weights is the same distribution as simulating walkers drawn from
    that mixture in one go. The two must agree to within sampling error.
    """
    lay = make_layers(seed=11, water=False)
    start = ipp_rc(lay)
    terrain = W.Terrain(lay)
    n = 3000
    cached = W.run_pure_strategies(terrain, start, n_per_strategy=n, times_h=TIMES, seed=5)

    mix = np.array([0.05, 0.37, 0.58, 0.0, 0.0, 0.0])
    mask = F.searchable(lay)
    # A direct run mixes walkers, so mix the cached counts by walker share.
    from_cache = F.normalize(np.tensordot(mix, cached[:, 0], axes=(0, 0)), mask)

    # A direct run with the same mixture, different walkers.
    sid = np.random.default_rng(9).choice(
        len(W.STRATEGIES), size=n, p=mix
    ).astype(np.int32)
    direct = np.zeros((G, G), np.float32)
    for i in range(len(W.STRATEGIES)):
        k = int((sid == i).sum())
        if k == 0:
            continue
        out = W.simulate(
            __import__("jax").random.PRNGKey(100 + i), terrain.as_arrays(),
            __import__("jax").numpy.full(k, i, __import__("jax").numpy.int32),
            np.int32(start[0]), np.int32(start[1]), k, 1.0, 0.0,
            G, TIMES, W.STEPS_PER_HOUR,
        )
        direct += np.asarray(out["hist"])[0]
    direct[0, :] = direct[-1, :] = 0.0          # same border rule as the cache
    direct[:, 0] = direct[:, -1] = 0.0
    from_direct = F.normalize(direct, mask)

    # Compared at the first slice (1 h), when nearly every walker is still
    # inside the 32 cell area. Total variation on 4 x 4 cell blocks.
    def pool(x):
        return x.reshape(G // 4, 4, G // 4, 4).sum((1, 3))
    tv = 0.5 * float(np.abs(pool(from_cache) - pool(from_direct)).sum())
    assert tv < 0.12, f"cached mixture and direct run differ, TV = {tv:.3f}"


def test_searchable_excludes_water():
    lay = make_layers(water=True)
    m = F.searchable(lay)
    assert not m[lay["water"]].any()
    assert not m[lay["blocked"]].any()
    assert m.any()


def test_time_weights_stay_inside_the_recorded_times():
    lo = service._time_weights(0.0)[0]
    assert lo[0] == 0
    hi = service._time_weights(1e6)[0]
    assert hi[1] == pytest.approx(1.0)
    mid = service._time_weights(3.0, (2.0, 4.0))
    assert sum(a for _, a in mid) == pytest.approx(1.0)


def test_load_params_falls_back_to_koester():
    """Plan 6.5: without a fitted file the baseline is the default."""
    params = service.load_params("runs/prior/definitely_missing.json")
    assert params["source"] == "fallback_koester"
    assert params["weights"]["walker"] == 0.0

# ------------------------------------------------------------ data plumbing

def test_tile_urls_cover_a_cut_across_tile_edges():
    """Case 39 straddles latitude 35 and case 8 sits in WorldCover N30W111."""
    from sar.prior import fetch

    b39 = fetch._bounds(34.97868334, -111.89643, 192, 30.0)
    dem = fetch._tile_urls(L.dem_url, b39, 1.0)
    assert any("N34_00_W112_00" in u for u in dem)
    assert any("N35_00_W112_00" in u for u in dem)
    wc8 = fetch._tile_urls(L.worldcover_url, fetch._bounds(32.0124, -109.3157, 192, 30.0), 3.0)
    assert wc8 == [L.WORLDCOVER_BUCKET_URL + "/ESA_WorldCover_10m_2021_v200_N30W111_Map.tif"]
    chm = L.canopy_url(32.0, -109.3)
    assert "meta_chm_lat=40.0_lon=-110.0_p95.tif" in chm


def test_worldcover_v200_codes_map_to_the_right_classes():
    wc = np.array([[10, 20, 30, 40, 50, 60], [70, 80, 90, 95, 100, 10]], np.uint8)
    lc = np.vectorize(lambda c: L.LC_OF_CLASS[int(c)])(wc)
    names = [[L.LC_NAMES[i] for i in row] for row in lc]
    assert names[0] == ["forest", "other", "grass", "grass", "built", "bare"]
    assert names[1] == ["bare", "water", "grass", "forest", "bare", "forest"]
    pd = L.pd_landcover(wc)
    assert pd.tolist() == [[2, 1, 1, 3, 4, 1], [1, 0, 1, 2, 1, 2]]


def test_osm_api_xml_is_parsed_like_overpass():
    from sar.prior import fetch

    xml = """<?xml version="1.0"?><osm>
      <node id="1" lat="31.0" lon="-111.0"/><node id="2" lat="31.001" lon="-111.0"/>
      <node id="3" lat="31.0" lon="-110.999"><tag k="amenity" v="trailhead"/></node>
      <way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="path"/></way>
      <way id="11"><nd ref="1"/><nd ref="3"/><tag k="landuse" v="farmland"/></way>
      <way id="12"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/>
        <tag k="building" v="yes"/><tag k="building:levels" v="3"/></way>
    </osm>"""
    els = fetch.parse_osm_xml(xml)
    kinds = sorted((e["type"], e["id"]) for e in els)
    assert kinds == [("node", 3), ("way", 10), ("way", 12)]
    way = next(e for e in els if e["id"] == 10)
    assert way["geometry"] == [{"lat": 31.0, "lon": -111.0}, {"lat": 31.001, "lon": -111.0}]
    assert fetch.building_height({"building:levels": "3"}) == pytest.approx(9.0)
    assert fetch.building_height({"height": "12.5 m"}) == pytest.approx(12.5)
    assert fetch.building_height({}) == pytest.approx(6.0)


def test_bank_round_trip_and_env_uses_matching_regions(tmp_path):
    """A written bank loads in the env layout, crops, and prior matches terrain."""
    import jax

    from sar.config_v2 import EnvConfigV2
    from sar.env.v2 import maps_v2
    from sar.env.v2.grid_env_v2 import SearchEnvV2

    ents, meta = [], []
    for i in range(3):
        lay = make_layers(seed=20 + i)
        lay["bheight"] = np.zeros((G, G), np.float32)
        start = ipp_rc(lay)
        prior = K.koester_map(lay, start)
        ent = service.bank_entry(lay, start, prior)
        ent["dem_m"] = ent["dem_m"] + 1000.0 * i          # tag each region
        ents.append(ent)
        meta.append({"name": f"r{i}", "lat": 40.0 + i, "lon": -110.0, "country": "US",
                     "group": "tile_x"})
    path = service.write_bank(ents, meta, str(tmp_path / "bank_train.npz"))
    full = maps_v2.load_prior_bank("train", tmp_path)
    assert full["landcover"].shape == (3, G, G)
    assert full["prior"].shape == (3, G, G)
    assert np.allclose(full["prior"].reshape(3, -1).sum(1), 1.0, atol=1e-5)
    assert (full["floor_m"] >= 10.0).all() and (full["floor_m"] < 79.0).all()
    crop = maps_v2.load_prior_bank("train", tmp_path, grid=16)
    assert crop["landcover"].shape == (3, 16, 16)
    assert np.allclose(crop["prior"].reshape(3, -1).sum(1), 1.0, atol=1e-5)
    assert path.endswith("bank_train.npz")

    cfg = EnvConfigV2(grid=G).with_grid(G)
    env = SearchEnvV2(cfg, prior_bank=full)
    for seed in range(3):
        s, _ = env.reset(jax.random.PRNGKey(seed), seed=seed)
        region = int(round(float(s.dem_m.min()) / 1000.0))
        assert region == seed % 3
        p = np.asarray(full["prior"][region]) * np.asarray(s.mask)
        b = np.asarray(s.belief)
        assert np.allclose(b / b.sum(), p / p.sum(), atol=1e-5)


