"""Fit the walker mixture and the blend weights on the real cases.

Plan sections 6.3 and 6.5. The prior for one case is

    prior = 0.05 * uniform + 0.95 * (a_K * koester + sum_s a_s * walker_s)

where `walker_s` is the end-point density of pure strategy `s` (RW, RT, DT,
SP, VE, BT) from the cached walker runs, and the `a` weights are on the
simplex. The prior is linear in `a`, so for a fixed walker grid point (trail
preference, stop probability) and time slice the maximum-likelihood weights
are found by EM over the find-cell values alone. Only the grid point and the
time slice are nonlinear, and they are picked by the training likelihood of
each fold.

Leave-one-out: for each case, the weights, grid point and time slice are fitted
on the other cases only, then the held-out case is scored on its full map.
Three variants are scored the same way:

    koester  0.05 uniform + 0.95 Koester                 (nothing fitted)
    walker   0.05 uniform + 0.95 walker strategy mix     (mix, grid, time fitted)
    blend    0.05 uniform + 0.95 (Koester + walker mix)  (all fitted)

The uniform floor here spreads over every cell of the cut, including water
and blocked cells, so a mapping error can never give a find zero probability.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import koester as K

# The 3 x 3 grid of the two nonlinear walker parameters (plan 6.3).
# `stop_prob` is the chance per hour that a walker stops for good.
TRAIL_PREF_GRID = (0.0, 1.0, 3.0)
STOP_PROB_GRID = (0.0, 0.05, 0.15)

# Gaussian smoothing of the walker end-point counts, in cells. 5000 walkers
# over a 192 x 192 cut leave most cells empty, so the raw counts give a find
# cell zero walker density by chance. The width is a third nonlinear
# parameter, picked inside each fold by training likelihood like the others.
SMOOTH_GRID = (0.0, 2.0, 4.0, 8.0, 12.0, 16.0)

UNIFORM_FLOOR = 0.05
OPERATOR_DEFAULT_WEIGHT = 0.3

EPS = 1e-12
N_STRAT = 6

# Component order in the value arrays: uniform, Koester, then the strategies.
COMP_UNIFORM, COMP_KOESTER = 0, 1


def normalize(x, mask):
    """Zero outside `mask` and scale to sum 1 inside it."""
    x = np.where(mask, np.maximum(np.asarray(x, np.float64), 0.0), 0.0)
    total = x.sum()
    if total <= 0:
        x = mask.astype(np.float64)
        total = max(x.sum(), 1.0)
    return x / total


def searchable(layers):
    """Cells the prior may put mass on (not water, not blocked)."""
    water = layers["lc"] == 4
    if "water" in layers:
        water = water | layers["water"]
    m = ~water
    if "blocked" in layers:
        m = m & ~layers["blocked"]
    return m


def smooth(x, sigma):
    """Gaussian blur of a count map, `sigma` in cells (0 leaves it as is)."""
    if not sigma:
        return np.asarray(x, np.float64)
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(np.asarray(x, np.float64), float(sigma), mode="constant")


def strategy_densities(cached_t, mask, sigma=0.0):
    """[6, g, g] counts of one time slice to six densities inside `mask`.

    The counts are blurred by `sigma` cells first. A strategy with no walker
    left inside the area falls back to uniform, so every component is a proper
    density.
    """
    return np.stack([normalize(smooth(cached_t[s], sigma), mask)
                     for s in range(cached_t.shape[0])])


def mix_maps(cached, strategy_mix, mask, sigma=0.0):
    """Blend cached pure-strategy maps with a strategy mixture.

    `cached` is [n_strategies, n_times, g, g] counts. Each strategy is first
    normalized on its own, so the mixture weights are shares of the walker
    density, the same quantity the fit estimates. Returns a list over time of
    [g, g] densities.
    """
    mix = np.asarray(strategy_mix, np.float64)
    mix = mix / max(mix.sum(), EPS)
    out = []
    for t in range(cached.shape[1]):
        dens = strategy_densities(cached[:, t], mask, sigma)
        out.append(normalize(np.tensordot(mix, dens, axes=(0, 0)), mask))
    return out


def component_maps(layers, ipp_rc, cached_t, koester=None, sigma=0.0):
    """[8, g, g] component densities for one case and one time slice."""
    mask = searchable(layers)
    g = mask.shape[0]
    uni = np.full((g, g), 1.0 / (g * g))
    ko = normalize(K.koester_map(layers, ipp_rc) if koester is None else koester, mask)
    return np.concatenate([uni[None], ko[None], strategy_densities(cached_t, mask, sigma)])


def em_weights(vals, allowed, floor=UNIFORM_FLOOR, iters=300):
    """Maximum-likelihood simplex weights over the non-uniform components.

    `vals` is [n_cases, n_comp] of each component's density at each case's
    find cell (column 0 is the uniform). The uniform weight is fixed at
    `floor` and the rest share `1 - floor`; `allowed` marks which of the other
    components may get weight. Returns the full weight vector (sums to 1).
    """
    vals = np.maximum(np.asarray(vals, np.float64), 0.0)
    n_comp = vals.shape[1]
    allowed = np.asarray(allowed, bool).copy()
    allowed[COMP_UNIFORM] = False
    a = allowed.astype(np.float64)
    a /= max(a.sum(), EPS)
    for _ in range(iters):
        p = floor * vals[:, COMP_UNIFORM] + (1.0 - floor) * vals @ a
        resp = (1.0 - floor) * vals * a[None, :] / np.maximum(p, EPS)[:, None]
        new = resp.sum(0)
        new = np.where(allowed, new, 0.0)
        if new.sum() <= 0:
            break
        new /= new.sum()
        if np.max(np.abs(new - a)) < 1e-7:
            a = new
            break
        a = new
    w = (1.0 - floor) * a
    w[COMP_UNIFORM] = floor
    assert w.shape == (n_comp,)
    return w


def loglik(vals, w):
    """Sum of log find-cell probabilities for weights `w`."""
    return float(np.sum(np.log(np.maximum(np.asarray(vals) @ w, EPS))))


def score(prior, find_rc, g=None):
    """Plan 6.5 metrics for one case: area share to reach the find, log-lik.

    The share is the fraction of the cut that a search visiting cells from
    most to least likely covers before it reaches the find cell (ties count
    against the map).
    """
    p = np.asarray(prior, np.float64)
    total = p.sum()
    if total <= 0:
        return 1.0, float(np.log(EPS))
    p = p / total
    cell = float(p[find_rc])
    share = float((p >= cell).sum()) / p.size
    return share, float(np.log(max(cell, EPS)))


def metrics(shares, lls):
    """Top 5/10/20 percent shares and the mean log-likelihood."""
    shares = np.asarray(shares, np.float64)
    if shares.size == 0:
        return {"top5": float("nan"), "top10": float("nan"), "top20": float("nan"),
                "mean_loglik": float("nan"), "median_share": float("nan"), "n": 0}
    return {
        "top5": float((shares <= 0.05).mean()),
        "top10": float((shares <= 0.10).mean()),
        "top20": float((shares <= 0.20).mean()),
        "mean_loglik": float(np.mean(lls)),
        "median_share": float(np.median(shares)),
        "n": int(shares.size),
    }


VARIANTS = {
    # name: which non-uniform components may get weight
    "koester": [False, True] + [False] * N_STRAT,
    "walker": [False, False] + [True] * N_STRAT,
    "blend": [False, True] + [True] * N_STRAT,
}


def fit_on(vals, train_idx, variant):
    """Fit one variant on the training cases.

    `vals` is [n_cases, n_grid, n_times, n_comp]. Returns (grid index, time
    index, weights, training log-likelihood). The Koester variant has nothing
    to fit and always uses weight 0.95 on Koester.
    """
    allowed = VARIANTS[variant]
    n_grid, n_t = vals.shape[1], vals.shape[2]
    best = None
    for gi in range(n_grid):
        for ti in range(n_t):
            v = vals[train_idx, gi, ti]
            w = em_weights(v, allowed)
            ll = loglik(v, w)
            if best is None or ll > best[3]:
                best = (gi, ti, w, ll)
            if variant == "koester":
                return best
    return best


def run_loo(vals, maps_fn, find_rc, verbose=False):
    """Leave-one-out over the cases.

    `vals` [n_cases, n_grid, n_times, n_comp] holds the find-cell values.
    `maps_fn(case, grid, time)` returns that case's [n_comp, g, g] component
    maps, used only to score the held-out case. `find_rc` is a list of cells.
    Returns per-variant metrics and the per-case rows.
    """
    n = vals.shape[0]
    rows = {k: [] for k in VARIANTS}
    for i in range(n):
        train = np.array([j for j in range(n) if j != i], int)
        for variant in VARIANTS:
            gi, ti, w, _ = fit_on(vals, train, variant)
            comp = maps_fn(i, gi, ti)
            prior = np.tensordot(w, comp, axes=(0, 0))
            share, ll = score(prior, tuple(find_rc[i]))
            rows[variant].append({"case": i, "share": share, "loglik": ll,
                                  "grid": int(gi), "time": int(ti),
                                  "weights": [float(x) for x in w]})
        if verbose:
            print(f"loo case {i}: " + " ".join(
                f"{k} {rows[k][-1]['share']:.3f}" for k in VARIANTS), flush=True)
    report = {k: metrics([r["share"] for r in rows[k]], [r["loglik"] for r in rows[k]])
              for k in VARIANTS}
    return report, rows


def fit_full(vals, variant="blend"):
    """Fit one variant on all cases. Returns (grid, time, weights, loglik)."""
    return fit_on(vals, np.arange(vals.shape[0]), variant)


def params_from_fit(gi, ti, w, times_h, n_cases, variant, ll):
    """The params dict `sar.prior.service.build_prior` reads."""
    w = np.asarray(w, np.float64)
    walker = float(w[2:].sum())
    mix = (w[2:] / walker) if walker > 0 else np.zeros(N_STRAT)
    tp, sp, sig = fit_grid()[int(gi)]
    return {
        "variant": variant,
        "trail_pref": tp,
        "stop_prob": sp,
        "smooth_cells": sig,
        "hours": float(times_h[ti]),
        "times_h": [float(t) for t in times_h],
        "weights": {"uniform": float(w[0]), "koester": float(w[1]), "walker": walker},
        "strategy_mix": [float(x) for x in mix],
        "loglik": float(ll),
        "n_cases": int(n_cases),
        "source": "fitted",
    }


def grid_points():
    """(trail_pref, stop_prob) pairs in the order the cache stores them."""
    return [(tp, sp) for tp in TRAIL_PREF_GRID for sp in STOP_PROB_GRID]


def fit_grid():
    """(trail_pref, stop_prob, smooth) triples; the fit's grid axis order.

    Index `gi` maps to walker cache point `gi // len(SMOOTH_GRID)` and
    smoothing `SMOOTH_GRID[gi % len(SMOOTH_GRID)]`.
    """
    return [(tp, sp, sg) for tp, sp in grid_points() for sg in SMOOTH_GRID]


def save_params(params, path="runs/prior/params.json"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path
