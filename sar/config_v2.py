"""Static configuration for the version 2 search environment (plan sections 7, 9.1).

Everything version 2 changes lives here. `sar/config.py` (version 1) is untouched so
its checkpoints in `runs/main_r4` keep loading and evaluating.

Changes against version 1 (plan section 7.1):
  - grid up to 192 cells, training curriculum 32 -> 64 -> 128, evaluation to 192
  - two altitude levels: low (1 cell footprint) and high (3 x 3 cells)
  - 66 actions: 32 ring-4 end cells or hold, times 2 levels
  - per-cell Pd from Pd[level, land cover class, canopy class]
  - battery sorties with a ground swap and relaunch
  - SIH-calibrated turn time 1.02 s per 45 degrees, plus climb and descend times
  - global view 32 x 32 at 12 cells per pixel (pooled coarser on small grids)
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

PD_TABLE_PATH = Path("data/pd_table.json")

# Land cover classes (indices into PdTable.pd) and canopy classes.
N_LANDCOVER = 5
LANDCOVER_NAMES = ("water", "grass", "forest", "cropland", "built")
N_CANOPY = 4
CANOPY_NAMES = ("none", "sparse", "dense", "tall")
N_LEVELS = 2
LOW, HIGH = 0, 1

# Area curriculum: grid side -> polygon extent range in cells (plan section 7.1)
STAGES_V2 = {32: (16, 30), 64: (30, 60), 128: (50, 126), 192: (60, 190)}


@dataclass(frozen=True)
class EnvConfigV2:
    # Grid and fleet
    grid: int = 128                # max grid side in cells (cell about 30 m)
    n_agents: int = 3              # max search drones
    ring: int = 4                  # segment length in cell moves (ring distance)
    crop: int = 31                 # local crop side
    global_size: int = 32          # global view side
    global_pool: int = 12          # cells per global view pixel (plan section 7.1)
    critic_pool: int = 4           # cells per critic map pixel (grid / 32)
    min_sep: int = 2               # cells within this Chebyshev distance are too close

    # Area curriculum: polygon extent in cells, sampled per episode
    extent_min: int = 50
    extent_max: int = 126

    # Timing in seconds. Turn time is the SIH-calibrated 1.02 s per 45 degrees
    # (plan section 7.1); straight and diagonal stay at the version 1 defaults.
    t_straight: float = 3.0
    t_diag: float = 4.24
    t_turn: float = 1.02           # per 45 degree heading change
    t_noise: float = 0.1           # relative time noise per move
    climb_rate: float = 3.0        # m/s; low-to-high and back is 46 m / 2 * this
    level_m: tuple = (23.0, 69.0)  # AGL of the two levels (plan section 9.1)
    cell_m: float = 30.0
    min_level_gap_m: float = 20.0  # height separation for crossing at different levels
    clearance_m: float = 10.0      # floor = ground + obstacle height + this (plan 9.1)

    # Battery and sorties
    battery_s: float = 1500.0      # flight time per drone, one sortie
    reserve_s: float = 60.0        # RTL reserve
    swap_s: float = 180.0          # battery swap on the ground (plan section 7.1)
    max_sorties: int = 4           # after this the drone stays out
    max_steps: int = 256           # decisions per episode

    # Detection model
    pd_scale: float = 1.0          # multiplies the table, for sensitivity tests
    pd_min_scale: float = 0.5      # per-episode multiplicative jitter range
    pd_max_scale: float = 1.5
    fp_max: float = 0.02
    freeze_k: int = 2              # decision steps frozen per candidate (re-look)

    # Fleet changes
    p_leave: float = 0.3           # prob a drone is recalled mid-episode
    p_join: float = 0.3            # prob drone 2/3 joins later instead of at start
    p_delay: float = 0.05          # random one-step delay per agent per step

    # Reward map mix and stopping
    mix_max: float = 0.5
    belief_stop: float = 0.02

    # Priors: 0 rings, 1 lines, 2 rings+lines, 3 blobs (held out), 4 prior bank
    train_prior_kinds: tuple = (0, 1, 2)
    eval_prior_kinds: tuple = (3,)

    # Altitude use: probability a segment keeps its level (else the planner may switch)
    p_level_switch: float = 0.0    # baseline keeps the level; greedy/look-ahead pick

    pd_table: object = None        # PdTable; loaded from data/pd_table.json if None

    @property
    def n_actions(self) -> int:
        """66 = 2 levels x (32 ring cells + hold)."""
        return N_LEVELS * (8 * self.ring + 1)

    @property
    def hold_action(self) -> int:
        """Hold at the low level; hold at the high level is `hold_action + 1`."""
        return 8 * self.ring

    def level(self, action: int) -> int:
        """Altitude level of an action: 0 low, 1 high."""
        return action // (8 * self.ring + 1)

    def base(self, action: int) -> int:
        """Ring action inside a level: 0..32, where 32 is hold."""
        return action % (8 * self.ring + 1)

    def with_grid(self, grid: int) -> "EnvConfigV2":
        lo, hi = STAGES_V2.get(grid, (grid // 2 - 14, grid - 2))
        return replace(self, grid=grid, extent_min=lo, extent_max=hi)


@dataclass(frozen=True)
class TrainConfigV2:
    """Expert iteration settings (plan section 8.3)."""

    seed: int = 0
    n_envs: int = 256
    rollout_len: int = 64
    lr: float = 2.5e-4
    gamma: float = 0.995
    max_grad_norm: float = 0.5
    sims_train: int = 16           # Gumbel MuZero simulations per decision while training
    sims_eval: int = 64            # simulations per decision at mission time
    max_considered: int = 16       # root sequential-halving fan-out
    value_scale: float = 0.1       # mctx qtransform value scale
    search_depth: int = 8          # mctx tree depth before the greedy rollout
    rollout_depth: int = 6         # greedy steps appended to each leaf value
    curriculum: tuple = field(default=((32, 32), (64, 64), (128, 128)))


def load_pd_table(path: Path | str = PD_TABLE_PATH) -> "PdTable":
    """Load Pd[level, land cover, canopy] from data/pd_table.json.

    The file is a placeholder until the detector gate of M1 measures recall per
    class on Isaac renders (plan section 10.3); see docs/DECISIONS.md.
    """
    from sar.env.v2.pd_table import PdTable

    return PdTable.load(Path(path))