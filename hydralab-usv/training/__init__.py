"""
HydraLab-USV training infrastructure.

Exports:
    train_ppo, make_vec_env     — SB3 PPO entry points
    DisturbanceCurriculum       — disturbance schedule
    make_curriculum_from_env    — factory from live env
    HydraLabMetricsCallback     — SB3 TensorBoard logger
    CurriculumCallback          — applies curriculum each step
    BestModelCallback           — saves best-reward checkpoint
    HydraLabRLlibEnv            — single-slot RLlib wrapper
    HydraLabRLlibVecEnv         — multi-slot RLlib wrapper
    StabilityValidator          — long-run numerical stability checker
    StabilityReport             — validation result dataclass
"""

from training.ppo_train import train_ppo, make_vec_env
from training.curriculum import DisturbanceCurriculum, make_curriculum_from_env
from training.callbacks import (
    HydraLabMetricsCallback,
    CurriculumCallback,
    BestModelCallback,
)
from training.rllib_env import HydraLabRLlibEnv, HydraLabRLlibVecEnv
from training.stability_validator import StabilityValidator, StabilityReport

__all__ = [
    "train_ppo",
    "make_vec_env",
    "DisturbanceCurriculum",
    "make_curriculum_from_env",
    "HydraLabMetricsCallback",
    "CurriculumCallback",
    "BestModelCallback",
    "HydraLabRLlibEnv",
    "HydraLabRLlibVecEnv",
    "StabilityValidator",
    "StabilityReport",
]
