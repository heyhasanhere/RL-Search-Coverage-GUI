"""Lost-person walker simulation in JAX (plan section 6.2).

Many walkers are simulated at once on the real terrain, each with one pure
strategy, and their positions are histogrammed at several elapsed times. One
run of a strategy therefore serves every time budget in the fit, and the
histograms for the pure strategies are cached on a 3 x 3 grid of the two
nonlinear parameters (trail preference, stop probability) so the mixture and
blend fit is linear in the cached maps.

Strategies (the six of Hashimoto et al. 2022)
---------------------------------------------
RW  random walk: heading wanders by a random turn each step
RT  route travel: follow trails and roads, pick a direction along the network
DT  direction travel: hold one heading and walk it out
SP  stay put: stop where they start
VE  view enhance: move uphill onto high ground and stop there
BT  backtrack: walk out, turn around and walk back

Walking speed is Tobler's hiking function on the slope. Water and cliffs block
a step. The start cell may be blocked (a last-seen point on a bridge); the
walker is pushed to the nearest free cell instead of being lost.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

STRATEGIES = ("RW", "RT", "DT", "SP", "VE", "BT")
STRATEGY_ID = {s: i for i, s in enumerate(STRATEGIES)}

# Neighbour offsets, ring 1 (8 cells) plus a hold. Order matters only for
# reproducibility; the same array is used by every step.
NEIGHBOUR_DC = jnp.array([0, 1, 1, 1, 0, -1, -1, -1, 0], jnp.int32)
NEIGHBOUR_DR = jnp.array([-1, -1, 0, 1, 1, 1, 0, -1, 0], jnp.int32)
NEIGHBOUR_COST = jnp.array(
    [1.0, 1.4142, 1.0, 1.4142, 1.0, 1.4142, 1.0, 1.4142, 0.0], jnp.float32
)

# Elapsed times in hours at which the end-point histogram is recorded.
DEFAULT_TIMES_H = (2.0, 4.0, 8.0, 16.0, 32.0)

# One step is one 30 m cell, taken with a chance set by Tobler's speed on the
# cell (0.84 on flat ground, about 0.26 on the median Arizona case slope).
# 20 steps an hour is 0.5 km/h on flat ground and about 0.15 km/h on that
# slope, so the time axis is "walking effort", not clock time; the fit picks
# the slice that best explains the finds (docs/DECISIONS.md).
STEPS_PER_HOUR = 20
TOBLER_MAX = 6.0 * 1.34 / 3.6

STEP_M = 30.0          # one grid step, so walkers move cell to cell
MAX_TRAIL_HOURS = 48.0


def tobler_speed(slope):
    """Tobler's hiking function in m/s for a signed slope (rise over run)."""
    k = 6.0 * jnp.exp(-3.5 * jnp.abs(slope + 0.05))
    return k * 1.34 / 3.6


class Terrain:
    """JAX-side terrain the walkers move on, built from a cut region."""

    def __init__(self, layers, cell_m=30.0):
        self.g = int(layers["lc"].shape[0])
        self.cell_m = float(cell_m)
        self.walk = jnp.asarray(layers["walk"], jnp.float32)
        self.blocked = jnp.asarray(layers["blocked"], jnp.bool_)
        self.slope = jnp.asarray(layers["slope"], jnp.float32)
        self.dem = jnp.asarray(layers["dem"], jnp.float32)
        self.trail = jnp.asarray(layers["trail"], jnp.float32)
        self.road = jnp.asarray(layers["road"], jnp.float32)
        self.stream = jnp.asarray(layers["stream"], jnp.float32)
        self.building = jnp.asarray(layers["building"], jnp.float32)
        self.step_time = jnp.asarray(layers.get("speed_penalty", 1.0), jnp.float32)
        self.free = (~self.blocked).astype(jnp.float32)
        self.pmove = jnp.clip(self.walk * self.step_time / TOBLER_MAX, 0.05, 1.0)
        # Elevation rank, for view enhance (higher is better).
        self.rank = _rank01(self.dem)
        # Trail preference per cell: trails first, then roads, then streams.
        route = jnp.maximum(
            jnp.maximum(self.trail, 0.9 * self.road), 0.4 * self.stream
        )
        self.route = route * self.free
        # Neighbour table for route travel: for each cell, the 8 directions
        # that are free, with the route weight of the target cell.
        self._build_route_graph()

    def _build_route_graph(self):
        """Neighbour tables for the 8 moves, used by route travel and view enhance."""
        dirs = list(zip(NEIGHBOUR_DR[:8], NEIGHBOUR_DC[:8]))
        self.nb_route = jnp.stack([_shift(self.route, dr, dc) for dr, dc in dirs])
        self.nb_free = jnp.stack([_shift(self.free, dr, dc) for dr, dc in dirs])
        cost = NEIGHBOUR_COST[:8, None, None]
        self.nb_step = jnp.stack(
            [_shift(self.walk * self.step_time, dr, dc) * cost for dr, dc in dirs]
        )
        self.nb_rank = jnp.stack([_shift(self.rank, dr, dc) for dr, dc in dirs])

    def as_arrays(self):
        """The pytree of arrays the jitted step needs.

        The grid size is not in here: it is a static argument so the Python
        loops over steps and directions compile to fixed-length unrolled code.
        """
        return {
            "free": self.free,
            "route": self.route,
            "nb_route": self.nb_route,
            "nb_free": self.nb_free,
            "nb_rank": self.nb_rank,
            "rank": self.rank,
            "pmove": self.pmove,
        }


def _shift(a, dr, dc):
    """Shift a grid by (dr, dc), padding with zeros outside."""
    out = jnp.zeros_like(a)
    gs = a.shape[0]
    r0, r1 = max(0, dr), gs + min(0, dr)
    c0, c1 = max(0, dc), gs + min(0, dc)
    out = out.at[r0:r1, c0:c1].set(a[r0 - dr:r1 - dr, c0 - dc:c1 - dc])
    return out


def _rank01(a):
    """Map an array to [0,1] by its rank, so heights compare without units."""
    flat = a.ravel()
    order = jnp.argsort(flat)
    ranks = jnp.zeros_like(order, jnp.float32).at[order].set(
        jnp.arange(flat.size, dtype=jnp.float32)
    )
    return (ranks / max(flat.size - 1, 1)).reshape(a.shape)


def nearest_free(free, r, c, g):
    """Push a start cell to the nearest free cell with a spiral search.

    A last-seen point can sit on a bridge or in a lake cell, so walkers start
    from the closest free cell within two cells. If nothing is free the start
    is kept, and the walker simply never moves.
    """
    gi = int(g)
    r = jnp.clip(r, 0, gi - 1)
    c = jnp.clip(c, 0, gi - 1)
    free_rc = free[r, c] > 0
    best = r * gi + c
    bestv = jnp.full(best.shape, 1e9, jnp.float32)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            rr = jnp.clip(r + dr, 0, gi - 1)
            cc = jnp.clip(c + dc, 0, gi - 1)
            ok = (free[rr, cc] > 0) & (dr * dr + dc * dc > 0)
            d = jnp.full(bestv.shape, dr * dr + dc * dc, jnp.float32)
            take = ok & (d < bestv)
            bestv = jnp.where(take, d, bestv)
            best = jnp.where(take, rr * gi + cc, best)
    out_r = jnp.where(free_rc, r, best // gi)
    out_c = jnp.where(free_rc, c, best % gi)
    return out_r, out_c


@functools.partial(
    jax.jit, static_argnames=("g", "times_h", "n_steps_per_hour", "n_walkers")
)
def simulate(key, terrain_arrays, strategy_ids, start_r, start_c, n_walkers,
             trail_pref, stop_prob, g, times_h=DEFAULT_TIMES_H,
             n_steps_per_hour=STEPS_PER_HOUR):
    """Public wrapper for the compiled walker step.

    `terrain_arrays` is a pytree of arrays (see `Terrain.as_arrays`). The grid
    size, walker count and time list are static, so each grid size compiles
    once and is then reused. `stop_prob` is the chance per hour that a walker
    stops for good (exhaustion, shelter, injury).
    """
    return _simulate(key, terrain_arrays, strategy_ids, start_r, start_c, n_walkers,
                     trail_pref, stop_prob, g, times_h, n_steps_per_hour)


def chunk_size_for(g, n_walkers):
    """Walkers per compiled call.

    The step is a `lax.scan` over steps, so the traced graph is one step
    regardless of the number of steps. The chunk bound only keeps the walker
    state small.
    """
    g = int(g)
    budget = 90000.0 / max(g, 1)
    return max(1024, int(min(n_walkers, budget)) // 256 * 256)


def _dir_affinity(dirs, heading):
    """How well each of the 8 directions lines up with a heading, in [-1, 1]."""
    diff = jnp.abs(((dirs - heading[None, :]) + 4) % 8 - 4)   # 0..4 steps of 45 deg
    return jnp.cos(diff.astype(jnp.float32) * (jnp.pi / 4.0))


def _simulate(key, terrain, strategy_ids, start_r, start_c, n_walkers,
              trail_pref, stop_prob, g, times_h, n_steps_per_hour):
    ks = jax.random.split(key, 8)
    n_t = len(times_h)
    # `times_h` is a static Python tuple, so these stay Python values.
    steps_at_py = [int(round(float(t) * n_steps_per_hour)) for t in times_h]
    steps_at = jnp.asarray(steps_at_py, jnp.int32)
    max_steps = max(steps_at_py) + 1
    # Per-step hazard that gives `stop_prob` per hour.
    hazard = 1.0 - jnp.power(1.0 - jnp.clip(stop_prob, 0.0, 0.999),
                             1.0 / float(n_steps_per_hour))

    r0, c0 = nearest_free(terrain["free"], start_r, start_c, g)
    r0 = jnp.full((n_walkers,), r0, jnp.int32)
    c0 = jnp.full((n_walkers,), c0, jnp.int32)
    nb_route = terrain["nb_route"]        # [8, g, g]
    nb_free = terrain["nb_free"]          # [8, g, g]
    nb_rank = terrain["nb_rank"]          # [8, g, g]
    rank = terrain["rank"]                # [g, g]
    free = terrain["free"]                # [g, g]
    pmove = terrain["pmove"]              # [g, g] Tobler speed as a move chance

    heading0 = jax.random.randint(ks[0], (n_walkers,), 0, 8)
    dt_head = jax.random.randint(ks[1], (n_walkers,), 0, 8)
    is_sp = strategy_ids == STRATEGY_ID["SP"]
    dirs = jnp.arange(8, dtype=jnp.int32)[:, None]
    rows = jnp.arange(n_walkers)
    half = max_steps // 2

    def step_fn(carry, step):
        r, c, heading, alive, hist = carry
        k = jax.random.fold_in(ks[2], step)
        k1, k2, k3, k4 = jax.random.split(k, 4)
        turn = jax.random.randint(k1, (n_walkers,), -1, 2)
        jit = jax.random.uniform(k2, (8, n_walkers))
        u = jax.random.uniform(k3, (2, n_walkers))

        frees = nb_free[:, r, c]           # [8, n]
        ok8 = frees > 0
        aff = _dir_affinity(dirs, heading)

        # Random walk: a correlated walk, the heading turns 45 degrees at most.
        rw_want = (heading + turn) % 8
        rw_score = _dir_affinity(dirs, rw_want) + 0.3 * jit
        # Route travel: follow trails and roads, keep going the same way.
        rt_score = trail_pref * nb_route[:, r, c] + 0.6 * aff + 0.3 * jit
        # Direction travel: hold one bearing, detour around what blocks it.
        dt_score = _dir_affinity(dirs, dt_head) + 0.1 * jit
        # View enhance: climb to a local high point and stop there.
        ranks = nb_rank[:, r, c]
        ve_score = ranks + 0.01 * jit
        # Backtrack: walk out, turn round once at half time, walk back.
        bt_want = jnp.where(step == half, (heading + 4) % 8, rw_want)
        bt_score = _dir_affinity(dirs, bt_want) + 0.3 * jit

        sid = strategy_ids[None, :]
        score = jnp.where(sid == STRATEGY_ID["RT"], rt_score, rw_score)
        score = jnp.where(sid == STRATEGY_ID["DT"], dt_score, score)
        score = jnp.where(sid == STRATEGY_ID["VE"], ve_score, score)
        score = jnp.where(sid == STRATEGY_ID["BT"], bt_score, score)
        score = jnp.where(ok8, score, -jnp.inf)
        move_dir = jnp.argmax(score, axis=0).astype(jnp.int32)
        any_free = jnp.any(ok8, axis=0)
        ve_done = (strategy_ids == STRATEGY_ID["VE"]) & (
            ranks[move_dir, rows] <= rank[r, c] + 1e-6)

        # Walking speed: a step is taken with a chance set by Tobler's speed
        # on this cell, and diagonal steps are 1.41 times longer.
        cost = jnp.take(NEIGHBOUR_COST, move_dir)
        go = u[0] < pmove[r, c] / jnp.maximum(cost, 1.0)
        alive = alive & (u[1] >= hazard)
        mover = (~is_sp) & alive & any_free & (~ve_done) & go
        nr = jnp.clip(r + jnp.take(NEIGHBOUR_DR, move_dir), 0, g - 1)
        nc = jnp.clip(c + jnp.take(NEIGHBOUR_DC, move_dir), 0, g - 1)
        ok = mover & (free[nr, nc] > 0)
        r = jnp.where(ok, nr, r)
        c = jnp.where(ok, nc, c)
        heading = jnp.where(ok, move_dir, heading)

        for ti in range(n_t):
            w = (steps_at[ti] == step).astype(jnp.float32)
            hist = hist.at[ti, r, c].add(w)
        return (r, c, heading, alive, hist), None

    init = (r0, c0, heading0, jnp.ones((n_walkers,), bool),
            jnp.zeros((n_t, g, g), jnp.float32))
    (r, c, _, _, hist), _ = jax.lax.scan(step_fn, init, jnp.arange(max_steps))
    return {"hist": hist, "r": r, "c": c}


def density(hist, mask=None):
    """Turn a histogram into a density in [0,1] summing to 1 inside `mask`."""
    h = hist.astype(jnp.float32)
    if mask is not None:
        h = jnp.where(mask, h, 0.0)
    return h / jnp.maximum(h.sum(), 1e-9)


def run_pure_strategies(terrain: Terrain, start_rc, n_per_strategy=4000,
                        trail_pref=1.0, stop_prob=0.0, times_h=DEFAULT_TIMES_H,
                        seed=0, n_steps_per_hour=STEPS_PER_HOUR):
    """Cached pure-strategy maps for one (trail_pref, stop_prob) grid point.

    Returns a numpy array [n_strategies, n_times, g, g] of raw counts, one
    plane per strategy and recorded time. Counts, not densities, so the caller
    can mix strategies by weight and normalize once.

    Walkers that reach the edge of the cut have left the area. Their counts on
    the border cells are dropped, so a long time budget does not pile mass on
    the frame. The share of walkers still inside is `counts.sum() / n`.
    """
    import numpy as np

    n_s = len(STRATEGIES)
    per_chunk = max(chunk_size_for(terrain.g, n_per_strategy), 64)
    total = n_per_strategy
    hist = np.zeros((n_s, len(times_h), terrain.g, terrain.g), np.float32)
    arrays = terrain.as_arrays()
    for i in range(n_s):
        done = 0
        while done < total:
            n = min(per_chunk, total - done)
            sid = jnp.full(n, i, jnp.int32)
            out = simulate(
                jax.random.fold_in(jax.random.PRNGKey(seed + 1000 * i), done),
                arrays, sid,
                jnp.int32(start_rc[0]), jnp.int32(start_rc[1]),
                n, float(trail_pref), float(stop_prob), terrain.g,
                tuple(float(t) for t in times_h), int(n_steps_per_hour),
            )
            hist[i] += np.asarray(out["hist"])
            done += n
    hist[:, :, 0, :] = 0.0
    hist[:, :, -1, :] = 0.0
    hist[:, :, :, 0] = 0.0
    hist[:, :, :, -1] = 0.0
    return hist
