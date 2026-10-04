"""Prior service: the one place the orchestrator and the env get a prior map.

`build_prior` takes the cut map layers, the last-seen point, the time since it
was seen and the fitted parameters, and returns a [H,W] float32 map that sums
to the expected survivor count. The env prior bank and the orchestrator both
call this, so a mission starts from the same map in both.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import fit as F
from . import koester as K
from . import walker as W
from .layers import DEFAULT_TREE_HEIGHT_M  # noqa: F401  (re-exported for callers)

DEFAULT_PARAMS_PATH = os.environ.get("SAR_PRIOR_PARAMS", "runs/prior/params.json")

# The walker's time axis is walking effort (see `walker.STEPS_PER_HOUR`), not
# clock time, and the 65 cases carry no time since last seen. The fit picks
# the effort slice that best explains the finds; a mission of this many hours
# since the last sighting is mapped onto that slice, and longer or shorter
# missions scale it linearly. This reference is an assumption, recorded in
# docs/DECISIONS.md.
REF_MISSION_HOURS = 12.0

DEFAULT_TIMES_H = W.DEFAULT_TIMES_H

# Plan 6.2 average mix over the 65 cases: 58 percent direction travel, 37
# percent route travel, 5 percent random walk. Used only when params carry no
# fitted mix.
DEFAULT_STRATEGY_MIX = np.array([0.05, 0.37, 0.58, 0.0, 0.0, 0.0])

MIN_HOURS = 0.25
MAX_HOURS = 72.0


def load_params(path=DEFAULT_PARAMS_PATH):
    """Load the fitted parameters, falling back to the Koester default.

    The plan (section 6.5) says the Koester baseline is the default when the
    blend does not beat it, so the fallback weights are the baseline with no
    walker term.
    """
    fallback = {
        "trail_pref": 1.0,
        "stop_prob": 0.0,
        "hours": 2.0,
        "weights": {"uniform": F.UNIFORM_FLOOR, "walker": 0.0,
                    "koester": 1.0 - F.UNIFORM_FLOOR},
        "loglik": None,
        "n_cases": 0,
        "source": "fallback_koester",
    }
    if not os.path.exists(path):
        return fallback
    with open(path) as f:
        params = json.load(f)
    params.setdefault("source", "fitted")
    return params


def effort_hours(hours_since_seen, params):
    """Walker effort slice for a mission `hours_since_seen` after the sighting."""
    base = float(params.get("hours", 2.0))
    if hours_since_seen is None:
        return base
    h = float(np.clip(hours_since_seen, MIN_HOURS, MAX_HOURS))
    return base * h / REF_MISSION_HOURS


def _time_weights(hours, times_h=DEFAULT_TIMES_H):
    """Linear weights over the two nearest recorded times."""
    t = np.asarray(times_h, np.float64)
    h = float(hours)
    if h <= t[0]:
        return [(0, 1.0)]
    if h >= t[-1]:
        return [(len(t) - 1, 1.0)]
    j = int(np.searchsorted(t, h))
    lo, hi = j - 1, j
    a = (h - t[lo]) / (t[hi] - t[lo])
    return [(lo, 1.0 - a), (hi, a)]


def run_walker(layers, ipp_rc, params, times_h, seed=0, n_per_strategy=4000):
    """Pure-strategy walker counts at the fitted grid point."""
    terrain = W.Terrain(layers, cell_m=float(layers.get("_cell_m", 30.0)))
    return W.run_pure_strategies(
        terrain, ipp_rc, n_per_strategy=n_per_strategy,
        trail_pref=float(params.get("trail_pref", 1.0)),
        stop_prob=float(params.get("stop_prob", 0.0)),
        times_h=tuple(float(t) for t in times_h), seed=seed,
    )


def build_prior(layers, ipp_rc, hours_since_seen, params=None,
                operator_regions=None, expected_count=1.0,
                seed=0, n_per_strategy=4000, times_h=None,
                walker_cache=None):
    """The prior map for one mission. Returns [H,W] float32.

    `layers` is a cut region dict (see `sar.prior.layers`). `ipp_rc` is the
    last-seen point as (row, col). `hours_since_seen` sets the walker effort
    (None uses the fitted slice). `operator_regions` is a list of boolean masks
    drawn by the operator, blended in with weight 0.3 (plan 6.3).
    `walker_cache` is [6, len(times_h), g, g] counts from an earlier run; when
    it is None the walker runs here for the one slice the mission needs.

    The result is zero on water and blocked cells and sums to
    `expected_count`.
    """
    if params is None:
        params = load_params()
    mask = F.searchable(layers)
    uni = F.normalize(mask.astype(np.float64), mask)
    ko = F.normalize(K.koester_map(layers, ipp_rc), mask)
    w = params["weights"]
    w_ko = float(w.get("koester", 0.0))
    w_w = float(w.get("walker", 0.0))
    w_u = float(w.get("uniform", F.UNIFORM_FLOOR))

    prior = w_ko * ko + w_u * uni
    if w_w > 1e-6:
        mix = np.asarray(params.get("strategy_mix", DEFAULT_STRATEGY_MIX), np.float64)
        effort = effort_hours(hours_since_seen, params)
        if walker_cache is None:
            times_h = (effort,)
            walker_cache = run_walker(layers, ipp_rc, params, times_h, seed=seed,
                                      n_per_strategy=n_per_strategy)
        elif times_h is None:
            times_h = params.get("times_h", DEFAULT_TIMES_H)
        by_time = F.mix_maps(walker_cache, mix, mask, params.get("smooth_cells", 0.0))
        walker = np.zeros_like(uni)
        for ti, a in _time_weights(effort, times_h):
            walker += a * by_time[ti]
        prior = prior + w_w * walker

    prior = F.normalize(prior, mask)
    if operator_regions:
        op = np.zeros_like(uni)
        for reg in operator_regions:
            op += F.normalize(np.asarray(reg, np.float64), mask)
        op = F.normalize(op, mask)
        prior = (1.0 - F.OPERATOR_DEFAULT_WEIGHT) * prior + F.OPERATOR_DEFAULT_WEIGHT * op
    prior = F.normalize(prior, mask)
    return (prior * expected_count).astype(np.float32)


def bank_entry(layers, ipp_rc, prior):
    """One region in the env v2 bank layout (see `layers.env_terrain`)."""
    from .layers import env_terrain

    ent = env_terrain(layers)
    ent["prior"] = np.asarray(prior, np.float32)
    ent["lc"] = np.asarray(layers["lc"], np.uint8)
    ent["canopy_m"] = np.asarray(layers["canopy"], np.float32)
    ent["blocked"] = np.asarray(layers["blocked"], bool)
    ent["ipp_rc"] = np.asarray(ipp_rc, np.int32)
    return ent


def write_bank(entries, meta, path):
    """Stack bank entries into the npz `sar.env.v2.maps_v2.load_prior_bank` reads.

    Arrays are stacked on a leading region axis: `landcover`, `canopy`,
    `dem_m`, `floor_m`, `prior` (what the env uses) plus `lc`, `canopy_m`,
    `blocked`, `ipp_rc` and the per-region `name`, `lat`, `lon`, `country`,
    `group`. All regions must have the same grid size.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = entries[0].keys()
    out = {k: np.stack([e[k] for e in entries]) for k in keys}
    for k in ("name", "country", "group"):
        out[k] = np.array([str(m.get(k, "")) for m in meta])
    for k in ("lat", "lon"):
        out[k] = np.array([float(m.get(k, np.nan)) for m in meta], np.float64)
    np.savez_compressed(path, **out)
    return path


def export_bank(regions, out_dir="data/regions", params=None, seed=0,
                expected_count=1.0, n_per_strategy=2000):
    """Build the prior for every region and write `bank_<split>.npz`.

    Each region dict needs `name`, `layers`, `ipp_rc`, `split` (train or
    heldout) and may carry `lat`, `lon`, `country`, `group` and
    `hours_since_seen`. Returns the paths written.
    """
    if params is None:
        params = load_params()
    splits = {}
    for i, r in enumerate(regions):
        prior = build_prior(
            r["layers"], r["ipp_rc"], r.get("hours_since_seen"), params,
            expected_count=expected_count, seed=seed + i,
            n_per_strategy=n_per_strategy,
        )
        ent = bank_entry(r["layers"], r["ipp_rc"], prior)
        splits.setdefault(r["split"], ([], []))
        splits[r["split"]][0].append(ent)
        splits[r["split"]][1].append({k: r.get(k, "") for k in
                                      ("name", "lat", "lon", "country", "group")})
    paths = []
    for split, (ents, meta) in splits.items():
        paths.append(write_bank(ents, meta, os.path.join(out_dir, f"bank_{split}.npz")))
    return paths, params
