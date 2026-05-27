"""
Disturbance curriculum scheduler.

Curriculum learning is critical for convergence on disturbance-rejection tasks.
Starting from full wind/current/waves from step 0 causes the policy to learn
reactive thrashing rather than smooth tracking.

Strategy
────────
    Phase A (warmup):   disturbance_scale = initial_scale   (policy finds baseline)
    Phase B (ramp):     scale increases linearly/cosine to final_scale
    Phase C (full):     disturbance_scale = final_scale   (maintain indefinitely)

The curriculum modifies **disturbance config objects in-place** on the live
environment, so changes take effect on the next episode reset without
rebuilding the environment.

Domain randomization is ramped separately — often on a slower schedule
than disturbances, so the policy sees a stable physics model before it
is randomized.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from configs.training_cfg import CurriculumConfig

if TYPE_CHECKING:
    from envs.blueboat_env import BlueboatEnv


class DisturbanceCurriculum:
    """Manages curriculum schedule and applies it to a running environment."""

    def __init__(
        self,
        cfg: CurriculumConfig,
        nominal_wind_mean: float,
        nominal_current_mean: float,
        nominal_wave_force_std: float,
        nominal_dr_fracs: dict[str, float],
    ) -> None:
        """
        Args:
            cfg:                     Curriculum configuration.
            nominal_wind_mean:       Full-difficulty wind speed mean (m/s).
            nominal_current_mean:    Full-difficulty current speed mean (m/s).
            nominal_wave_force_std:  Full-difficulty wave force std (N).
            nominal_dr_fracs:        Dict of nominal DR fractions by param name
                                     (mass_noise_frac, drag_noise_frac, etc.).
        """
        self.cfg = cfg
        self._wind_nom = nominal_wind_mean
        self._cur_nom = nominal_current_mean
        self._wave_nom = nominal_wave_force_std
        self._dr_nom = nominal_dr_fracs

        # Use -1.0 as a sentinel so the very first apply() always writes to the env,
        # even if the computed scale matches initial_disturbance_scale.
        self._last_scale: float = -1.0
        self._last_dr_scale: float = -1.0

    # ------------------------------------------------------------------
    # Scale computation
    # ------------------------------------------------------------------

    def get_disturbance_scale(self, step: int) -> float:
        """Return disturbance scale ∈ [initial, final] at `step`."""
        return self._compute_scale(
            step,
            self.cfg.initial_disturbance_scale,
            self.cfg.final_disturbance_scale,
        )

    def get_dr_scale(self, step: int) -> float:
        """Return domain randomization scale ∈ [initial_dr, final_dr] at `step`."""
        return self._compute_scale(
            step,
            self.cfg.initial_dr_scale,
            self.cfg.final_dr_scale,
        )

    def _compute_scale(
        self,
        step: int,
        lo: float,
        hi: float,
    ) -> float:
        cfg = self.cfg

        if not cfg.enabled:
            return hi

        if step < cfg.warmup_steps:
            return lo
        if step >= cfg.full_steps:
            return hi

        progress = (step - cfg.warmup_steps) / max(
            cfg.full_steps - cfg.warmup_steps, 1
        )

        if cfg.schedule == "linear":
            alpha = progress
        elif cfg.schedule == "cosine":
            alpha = (1.0 - math.cos(math.pi * progress)) / 2.0
        else:  # step
            alpha = lo
            for boundary, val in zip(cfg.step_boundaries, cfg.step_values):
                if progress >= boundary:
                    alpha = val
            return lo + alpha * (hi - lo)

        return lo + alpha * (hi - lo)

    # ------------------------------------------------------------------
    # Apply to environment
    # ------------------------------------------------------------------

    def apply(self, env: "BlueboatEnv", step: int) -> None:
        """Modify env disturbance and DR config in-place for `step`.

        This is called each training step by CurriculumCallback.
        Only updates when the scale changes to avoid redundant writes.

        Args:
            env:  Live BlueboatEnv instance.
            step: Current number of environment steps.
        """
        dist_scale = self.get_disturbance_scale(step)
        dr_scale = self.get_dr_scale(step)

        if dist_scale != self._last_scale:
            self._apply_disturbance_scale(env, dist_scale)
            self._last_scale = dist_scale

        if dr_scale != self._last_dr_scale:
            self._apply_dr_scale(env, dr_scale)
            self._last_dr_scale = dr_scale

    def _apply_disturbance_scale(self, env: "BlueboatEnv", scale: float) -> None:
        """Scale disturbance magnitudes on the live environment config."""
        dc = env.cfg.disturbance

        dc.wind_speed_mean = scale * self._wind_nom
        dc.wind_speed_std = scale * self._wind_nom * 0.25

        dc.current_speed_mean = scale * self._cur_nom
        dc.current_speed_std = scale * self._cur_nom * 0.33

        dc.wave_force_std = scale * self._wave_nom
        dc.wave_moment_std = scale * self._wave_nom * 0.3

        # Propagate to live disturbance objects (affects newly-reset episodes)
        env.wind.cfg.wind_speed_mean = dc.wind_speed_mean
        env.wind.cfg.wind_speed_std = dc.wind_speed_std
        env.current.cfg.current_speed_mean = dc.current_speed_mean
        env.current.cfg.current_speed_std = dc.current_speed_std

    def _apply_dr_scale(self, env: "BlueboatEnv", scale: float) -> None:
        """Scale domain randomization noise fractions."""
        if not env.cfg.randomization.enabled:
            return
        rc = env.cfg.randomization
        nom = self._dr_nom

        rc.mass_noise_frac = scale * nom.get("mass_noise_frac", 0.10)
        rc.drag_noise_frac = scale * nom.get("drag_noise_frac", 0.20)
        rc.inertia_noise_frac = scale * nom.get("inertia_noise_frac", 0.10)

        lo, hi = nom.get("thrust_lo", 0.8), nom.get("thrust_hi", 1.0)
        # Compress range toward 1.0 at low scale
        adjusted_lo = 1.0 - scale * (1.0 - lo)
        rc.thrust_efficiency_range = (adjusted_lo, hi)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self, step: int) -> dict[str, float]:
        """Return a dict of current curriculum state for logging."""
        return {
            "curriculum/disturbance_scale": self.get_disturbance_scale(step),
            "curriculum/dr_scale": self.get_dr_scale(step),
            "curriculum/progress": min(
                1.0,
                max(0.0, (step - self.cfg.warmup_steps)
                    / max(self.cfg.full_steps - self.cfg.warmup_steps, 1)),
            ),
        }


def make_curriculum_from_env(
    env: "BlueboatEnv",
    curriculum_cfg: CurriculumConfig,
) -> DisturbanceCurriculum:
    """Construct a curriculum from a live environment's nominal config.

    Reads nominal (full-difficulty) disturbance values from the env config.
    """
    dc = env.cfg.disturbance
    rc = env.cfg.randomization

    return DisturbanceCurriculum(
        cfg=curriculum_cfg,
        nominal_wind_mean=dc.wind_speed_mean,
        nominal_current_mean=dc.current_speed_mean,
        nominal_wave_force_std=dc.wave_force_std,
        nominal_dr_fracs={
            "mass_noise_frac": rc.mass_noise_frac,
            "drag_noise_frac": rc.drag_noise_frac,
            "inertia_noise_frac": rc.inertia_noise_frac,
            "thrust_lo": rc.thrust_efficiency_range[0],
            "thrust_hi": rc.thrust_efficiency_range[1],
        },
    )
