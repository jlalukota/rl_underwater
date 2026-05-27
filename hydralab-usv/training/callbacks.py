"""
Custom Stable-Baselines3 training callbacks for HydraLab-USV.

HydraLabMetricsCallback
    Logs marine-specific diagnostics to TensorBoard at each rollout collection:
        - mean distance to goal
        - mean surge velocity
        - mean wind / current magnitudes
        - episode return / length / tracking error (from EpisodeStats)
        - action statistics (saturation fraction)

CurriculumCallback
    Applies the DisturbanceCurriculum to the live environment every N steps.
    Logs the current curriculum scale and progress.

Usage in training script:
    curriculum = make_curriculum_from_env(env, training_cfg.curriculum)
    callbacks = [
        HydraLabMetricsCallback(env, verbose=1),
        CurriculumCallback(curriculum, env, update_freq=256),
    ]
    model.learn(total_timesteps=..., callback=callbacks)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    # Provide a minimal stub so imports don't crash without SB3
    class BaseCallback:  # type: ignore
        def __init__(self, verbose=0): self.verbose = verbose
        def _on_step(self): return True

if TYPE_CHECKING:
    from envs.blueboat_env import BlueboatEnv
    from training.curriculum import DisturbanceCurriculum


# ---------------------------------------------------------------------------
# Marine metrics logger
# ---------------------------------------------------------------------------

class HydraLabMetricsCallback(BaseCallback):
    """TensorBoard logger for marine-specific training diagnostics.

    Intended to be used alongside VecMonitor — this callback adds *additional*
    environment-specific metrics that VecMonitor doesn't capture.
    """

    def __init__(
        self,
        env: "BlueboatEnv",
        log_freq: int = 1,
        verbose: int = 0,
    ) -> None:
        """
        Args:
            env:       The BlueboatEnv instance (unwrapped).
            log_freq:  Log every N rollout collections.
            verbose:   SB3 verbosity level.
        """
        super().__init__(verbose)
        self._env = env
        self._log_freq = log_freq
        self._n_calls_local = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._n_calls_local += 1
        if self._n_calls_local % self._log_freq != 0:
            return

        env = self._env

        # --- Distance to goal ---
        dist = env.r_tracking.distance(
            env.body.pos, env.task.target_pos
        ).mean().item()
        self.logger.record("marine/mean_dist_to_goal_m", dist)

        # --- Velocities ---
        self.logger.record("marine/mean_surge_mps", env.body.surge.mean().item())
        self.logger.record("marine/mean_sway_mps", env.body.sway.mean().item())
        self.logger.record("marine/mean_yaw_rate_rads", env.body.yaw_rate.mean().item())

        # --- Disturbances ---
        self.logger.record("marine/wind_speed_mean", env.wind.wind_speed.mean().item())
        self.logger.record("marine/current_speed_mean", env.current.speed.mean().item())

        # --- Action statistics ---
        # Saturation fraction: fraction of thrusters at ±1.0 (clipped)
        import torch
        a = env.prev_action
        sat_frac = (a.abs() > 0.98).float().mean().item()
        self.logger.record("marine/action_saturation_frac", sat_frac)

        # --- Episode statistics from EpisodeStats ---
        summary = env.stats.summary(last_n=env.num_envs)
        if summary:
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    self.logger.record(f"marine/ep_{k}", v)

        # --- Boundary violations ---
        boundary_frac = env.r_boundary.is_violated(env.body.pos).float().mean().item()
        self.logger.record("marine/boundary_violation_frac", boundary_frac)


# ---------------------------------------------------------------------------
# Curriculum applier
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """Applies disturbance curriculum schedule to the live environment.

    Calls `curriculum.apply(env, num_timesteps)` every `update_freq` steps
    and logs current scale to TensorBoard.
    """

    def __init__(
        self,
        curriculum: "DisturbanceCurriculum",
        env: "BlueboatEnv",
        update_freq: int = 1024,
        verbose: int = 0,
    ) -> None:
        """
        Args:
            curriculum:  DisturbanceCurriculum instance.
            env:         The live BlueboatEnv (unwrapped).
            update_freq: Apply curriculum every N env steps.
            verbose:     SB3 verbosity.
        """
        super().__init__(verbose)
        self._curriculum = curriculum
        self._env = env
        self._update_freq = update_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self._update_freq < self.training_env.num_envs:
            self._curriculum.apply(self._env, self.num_timesteps)

            # Log current curriculum state
            summary = self._curriculum.summary(self.num_timesteps)
            for k, v in summary.items():
                self.logger.record(k, v)

        return True


# ---------------------------------------------------------------------------
# Best-model checkpoint callback
# ---------------------------------------------------------------------------

class BestModelCallback(BaseCallback):
    """Save the model whenever mean episode return improves.

    Complements CheckpointCallback (which saves on a fixed schedule)
    by also tracking the best-performing checkpoint.
    """

    def __init__(
        self,
        save_path: str,
        check_freq: int = 10_000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self._best_mean_reward = float("-inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        # Access episode info from monitor wrapper
        if hasattr(self.training_env, "get_episode_rewards"):
            ep_rewards = self.training_env.get_episode_rewards()
            if len(ep_rewards) > 0:
                mean_reward = float(np.mean(ep_rewards[-100:]))
                if mean_reward > self._best_mean_reward:
                    self._best_mean_reward = mean_reward
                    self.model.save(f"{self.save_path}/best_model")
                    if self.verbose >= 1:
                        print(
                            f"[BestModel] New best: {mean_reward:.2f}  "
                            f"(step {self.num_timesteps:,})"
                        )
        return True
