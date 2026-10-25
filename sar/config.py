"""Static configuration for the grid search environment and training.

All tunable numbers live here. Values marked "default timing" are replaced
after PX4 SIH calibration (plan stage 1b).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvConfig:
    # Grid and fleet
    grid: int = 96                 # max grid side in cells (cell about 30 m)
    n_agents: int = 3              # max search drones
    ring: int = 4                  # segment length in cell moves (ring distance)
    crop: int = 31                 # local crop side
    global_size: int = 32          # global view side
    global_pool: int = 6           # cells per global view pixel
    critic_pool: int = 3           # cells per critic map pixel (grid / 32)
    min_sep: int = 2               # cells within this Chebyshev distance are too close

    # Area curriculum: polygon extent in cells, sampled per episode
    extent_min: int = 24
    extent_max: int = 32

    # Default timing in seconds (replaced by SIH calibration)
    t_straight: float = 3.0
    t_diag: float = 4.24
    t_turn: float = 0.75           # per 45 degree heading change
    t_noise: float = 0.1           # relative time noise per move
    battery_s: float = 1500.0      # flight time per drone
    reserve_s: float = 60.0        # RTL reserve
    max_steps: int = 256           # decisions per episode

    # Detection model
    pd_min: float = 0.6
    pd_max: float = 0.95
    fp_max: float = 0.02
    freeze_k: int = 2              # decision steps frozen per candidate (re-look)

    # Fleet changes
    p_leave: float = 0.3           # prob a drone is recalled mid-episode
    p_join: float = 0.3            # prob drone 2/3 joins later instead of at start
    p_delay: float = 0.05          # random one-step delay per agent per step

    # Reward map mix and stopping
    mix_max: float = 0.5
    belief_stop: float = 0.02

    # Priors: 0 rings, 1 lines, 2 rings+lines, 3 blobs (held out)
    train_prior_kinds: tuple = (0, 1, 2)
    eval_prior_kinds: tuple = (3,)

    @property
    def n_actions(self) -> int:
        return 8 * self.ring + 1


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    n_seeds: int = 4
    n_envs: int = 256
    rollout_len: int = 64
    # Behavior cloning
    bc_iters: int = 300
    bc_epochs: int = 2
    bc_minibatches: int = 8
    bc_lr: float = 3e-4
    bc_beta_end: float = 0.5        # final prob of executing the student (DAgger)
    # MAPPO
    ppo_updates: int = 2000
    ppo_epochs: int = 4
    ppo_minibatches: int = 8
    lr: float = 2.5e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    critic_warmup: int = 50         # updates with actor frozen
    kl_coef: float = 0.5            # KL to cloned policy, fades to 0
    kl_fade: int = 500              # updates until KL weight reaches 0
    curriculum: tuple = field(default=((24, 32), (40, 64), (64, 96)))


@dataclass(frozen=True)
class MissionConfig:
    """Orchestrator settings (plan section 4). Timing matches EnvConfig defaults."""
    home: tuple = (149.58, -33.42)    # [lon, lat] of the launch site and sim world origin
    cell_m: float = 30.0
    grid: int = 96
    search_alt: float = 23.0           # m AGL: 1.15 rad camera footprint is one 30 m cell
    transit_alt: tuple = (30.0, 35.0, 40.0)   # per drone, for transit and RTL
    cruise: float = 10.0               # m/s along search segments (30 m cell in 3 s)
    transit_speed: float = 12.0
    climb: float = 3.0                 # m/s
    accel: float = 2.5                 # m/s2: planned speed-up and braking along a path, under what the drone can do
    turn_speed: float = 5.0            # m/s through a right-angle turn; a U-turn stops, a straight joint keeps speed
    tick: float = 0.2                  # orchestrator tick in sim seconds
    battery_s: float = 1500.0          # flight time per drone
    reserve_s: float = 60.0
    land_s: float = 15.0               # landing time added to the return estimate
    link_timeout: float = 3.0          # s without telemetry before ABORT
    pd: float = 0.8                    # detection probability used by the belief update
    merge_m: float = 8.0               # detections closer than this are the same survivor
    drone_mask_m: float = 4.0          # detections this close to a lower drone are that drone, not a person
    relook_offset: float = 8.0         # displaced viewpoint for the persistence check
    relook_hold: float = 2.0           # s hovering at the re-look point
    rescue_hold: float = 20.0          # s over a survivor before the rescue drone is free
    home_spacing: float = 6.0          # m between drone home pads
    stage_period: float = 30.0         # s between auto-stage updates
