"""
Training and curriculum configuration dataclasses.

Separated from blueboat_cfg.py to keep environment physics config
decoupled from training infrastructure config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CurriculumConfig:
    """Disturbance curriculum schedule.

    Ramps disturbance magnitude from `initial_scale` to `final_scale` over
    training.  Domain randomization magnitude is ramped separately.

    Schedule shapes:
        linear  — constant ramp rate
        cosine  — slow start, faster mid, slow finish (smoother gradient)
        step    — discrete jumps at `step_boundaries` fractions of total steps
    """

    enabled: bool = True
    schedule: Literal["linear", "cosine", "step"] = "linear"

    # Steps before ramping begins (let policy find a basic solution first)
    warmup_steps: int = 500_000
    # Steps at which full difficulty is reached
    full_steps: int = 5_000_000

    # Disturbance scale range
    initial_disturbance_scale: float = 0.05   # near-zero at start
    final_disturbance_scale: float = 1.0

    # Domain randomization scale range (DR disabled initially)
    initial_dr_scale: float = 0.0
    final_dr_scale: float = 1.0

    # For "step" schedule: fraction of total training at each jump
    step_boundaries: tuple[float, ...] = (0.25, 0.5, 0.75)
    step_values: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


@dataclass
class PolicyConfig:
    """Neural network policy architecture."""

    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    activation: Literal["tanh", "relu", "elu"] = "tanh"
    # Whether to use a shared feature extractor for actor and critic
    shared_features: bool = True
    # Optional: use LSTM for partial observability (adds obs history dependency)
    use_lstm: bool = False
    lstm_hidden_size: int = 256


@dataclass
class TrainingConfig:
    """Hyperparameters for PPO training."""

    # --- Core PPO ---
    total_timesteps: int = 50_000_000
    n_steps: int = 256            # rollout steps per env per update iteration
    batch_size: int = 4096        # minibatch size
    n_epochs: int = 10            # update epochs per rollout
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None   # None = same as clip_range
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    # --- Logging & checkpointing ---
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 500_000
    log_interval: int = 10       # log every N update iterations
    eval_freq: int = 1_000_000   # evaluate every N timesteps
    n_eval_episodes: int = 50

    # --- Reproducibility ---
    seed: int = 42

    # --- Sub-configs ---
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
