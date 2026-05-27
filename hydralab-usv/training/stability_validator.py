"""
Numerical stability validator for HydraLab-USV.

Runs the environment for N steps under stress conditions (random actions,
full disturbances, maximum thrust) and checks for:
    - NaN / Inf in state, observations, or rewards
    - Position explosion beyond physical workspace
    - Velocity explosion beyond clamped limits
    - Reward divergence (extremely large magnitudes)

Usage:
    from training.stability_validator import StabilityValidator

    report = StabilityValidator(n_steps=50_000, n_envs=64).run()
    if not report.passed:
        print(report.summary())
        sys.exit(1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from configs.blueboat_cfg import HydraLabConfig


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class StabilityReport:
    """Results of a stability validation run."""

    passed: bool
    n_steps: int
    failure_step: Optional[int] = None
    reason: str = ""

    # Aggregate statistics
    max_pos_norm: float = 0.0
    max_vel_norm: float = 0.0
    max_reward_abs: float = 0.0
    nan_obs_count: int = 0
    inf_obs_count: int = 0

    def summary(self) -> str:
        lines = [
            f"StabilityReport({'PASS' if self.passed else 'FAIL'})",
            f"  steps run      : {self.failure_step if self.failure_step is not None else self.n_steps}",
            f"  max |pos|      : {self.max_pos_norm:.2f} m",
            f"  max |vel|      : {self.max_vel_norm:.4f} m/s (or rad/s)",
            f"  max |reward|   : {self.max_reward_abs:.4f}",
            f"  NaN obs        : {self.nan_obs_count}",
            f"  Inf obs        : {self.inf_obs_count}",
        ]
        if not self.passed:
            lines.append(f"  FAILURE        : {self.reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class StabilityValidator:
    """Long-run numerical stability checker for BlueboatEnv.

    Args:
        cfg:          HydraLabConfig. Uses default (full disturbances) if None.
        n_steps:      Number of policy steps to simulate.
        n_envs:       Number of parallel environments.
        device:       Torch device ("cpu" or "cuda").
        action_mode:  "random"  — uniform random actions
                      "max"     — always maximum thrust (stress test)
                      "zero"    — no thrust (free drift)
        max_pos_m:    Fail if any vessel exceeds this distance from origin.
        max_vel:      Fail if any velocity component exceeds this (m/s, rad/s).
        max_rew_abs:  Fail if any single-step reward magnitude exceeds this.
    """

    def __init__(
        self,
        cfg: Optional[HydraLabConfig] = None,
        n_steps: int = 10_000,
        n_envs: int = 64,
        device: str = "cpu",
        action_mode: str = "random",
        max_pos_m: float = 500.0,
        max_vel: float = 50.0,
        max_rew_abs: float = 1e4,
    ) -> None:
        self.cfg = cfg or self._make_stress_cfg(n_envs, device)
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.action_mode = action_mode
        self.max_pos_m = max_pos_m
        self.max_vel = max_vel
        self.max_rew_abs = max_rew_abs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> StabilityReport:
        """Run the validation loop and return a StabilityReport."""
        from envs.blueboat_env import BlueboatEnv

        env = BlueboatEnv(self.cfg)
        env.reset()

        max_pos = 0.0
        max_vel = 0.0
        max_rew = 0.0
        nan_count = 0
        inf_count = 0

        for step in range(self.n_steps):
            actions = self._sample_actions()
            obs, rew, terminated, truncated, _ = env.step(actions)

            # --- NaN / Inf check ---
            nan_now = obs.isnan().sum().item()
            inf_now = obs.isinf().sum().item()
            nan_count += int(nan_now)
            inf_count += int(inf_now)

            if nan_now > 0:
                return StabilityReport(
                    passed=False,
                    n_steps=self.n_steps,
                    failure_step=step,
                    reason=f"NaN in observations at step {step}",
                    max_pos_norm=max_pos,
                    max_vel_norm=max_vel,
                    max_reward_abs=max_rew,
                    nan_obs_count=nan_count,
                    inf_obs_count=inf_count,
                )
            if inf_now > 0:
                return StabilityReport(
                    passed=False,
                    n_steps=self.n_steps,
                    failure_step=step,
                    reason=f"Inf in observations at step {step}",
                    max_pos_norm=max_pos,
                    max_vel_norm=max_vel,
                    max_reward_abs=max_rew,
                    nan_obs_count=nan_count,
                    inf_obs_count=inf_count,
                )

            # --- State bound check ---
            state = env.body.state
            pos_norm = state[:, :2].norm(dim=-1).max().item()
            vel_norm = state[:, 3:].abs().max().item()
            rew_abs = rew.abs().max().item()

            max_pos = max(max_pos, pos_norm)
            max_vel = max(max_vel, vel_norm)
            max_rew = max(max_rew, rew_abs)

            if pos_norm > self.max_pos_m:
                return StabilityReport(
                    passed=False,
                    n_steps=self.n_steps,
                    failure_step=step,
                    reason=(
                        f"Position explosion: |pos|={pos_norm:.1f} m "
                        f"exceeds limit {self.max_pos_m} m at step {step}"
                    ),
                    max_pos_norm=max_pos,
                    max_vel_norm=max_vel,
                    max_reward_abs=max_rew,
                    nan_obs_count=nan_count,
                    inf_obs_count=inf_count,
                )

            if vel_norm > self.max_vel:
                return StabilityReport(
                    passed=False,
                    n_steps=self.n_steps,
                    failure_step=step,
                    reason=(
                        f"Velocity explosion: max vel={vel_norm:.2f} "
                        f"exceeds limit {self.max_vel} at step {step}"
                    ),
                    max_pos_norm=max_pos,
                    max_vel_norm=max_vel,
                    max_reward_abs=max_rew,
                    nan_obs_count=nan_count,
                    inf_obs_count=inf_count,
                )

            if not math.isfinite(rew_abs) or rew_abs > self.max_rew_abs:
                return StabilityReport(
                    passed=False,
                    n_steps=self.n_steps,
                    failure_step=step,
                    reason=(
                        f"Reward divergence: |r|={rew_abs:.2f} "
                        f"exceeds limit {self.max_rew_abs} at step {step}"
                    ),
                    max_pos_norm=max_pos,
                    max_vel_norm=max_vel,
                    max_reward_abs=max_rew,
                    nan_obs_count=nan_count,
                    inf_obs_count=inf_count,
                )

        return StabilityReport(
            passed=True,
            n_steps=self.n_steps,
            max_pos_norm=max_pos,
            max_vel_norm=max_vel,
            max_reward_abs=max_rew,
            nan_obs_count=nan_count,
            inf_obs_count=inf_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_actions(self) -> "torch.Tensor":
        if self.action_mode == "random":
            return torch.rand(self.n_envs, 2, device=self.device) * 2 - 1
        elif self.action_mode == "max":
            return torch.ones(self.n_envs, 2, device=self.device)
        else:  # "zero"
            return torch.zeros(self.n_envs, 2, device=self.device)

    @staticmethod
    def _make_stress_cfg(n_envs: int, device: str) -> HydraLabConfig:
        """Create a config with full disturbances for stress testing."""
        cfg = HydraLabConfig()
        cfg.env.num_envs = n_envs
        cfg.device = device
        # Full disturbances from step 0 (no curriculum warm-up)
        cfg.disturbance.wind_speed_mean = 10.0
        cfg.disturbance.wind_speed_std = 2.5
        cfg.disturbance.current_speed_mean = 0.5
        cfg.disturbance.current_speed_std = 0.15
        cfg.disturbance.wave_force_std = 8.0
        return cfg
