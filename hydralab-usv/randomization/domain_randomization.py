"""
Domain randomization pipeline for HydraLab-USV.

Applied at episode reset, this module:
    1. Generates randomised initial vessel states
    2. Randomises disturbance parameters (wind speed/dir, current)
    3. Randomises physical parameters (mass, drag, thruster efficiency)

All randomization is GPU-accelerated via PyTorch — no Python loops per env.

Design goal: sim-to-real transfer.  Every parameter that is uncertain
in the real system should be randomised here with a range that brackets
the real-world value.
"""

from __future__ import annotations

import torch
from torch import Tensor

from configs.blueboat_cfg import DomainRandomizationConfig, EnvironmentConfig


class DomainRandomizer:
    """Generates randomised initial conditions and perturbs model parameters."""

    def __init__(
        self,
        dr_cfg: DomainRandomizationConfig,
        env_cfg: EnvironmentConfig,
        num_envs: int,
        device: str = "cuda",
    ) -> None:
        self.dr_cfg = dr_cfg
        self.env_cfg = env_cfg
        self.num_envs = num_envs
        self.device = device

    # ------------------------------------------------------------------
    # Initial state generation
    # ------------------------------------------------------------------

    def sample_initial_states(self, env_ids: Tensor) -> Tensor:
        """Sample random initial 6-DOF states for specified environments.

        Returns:
            init_state: (len(env_ids), 6)  [x, y, psi, u, v, r]
        """
        n = env_ids.numel()
        cfg = self.dr_cfg

        # Position: uniform scatter around origin
        xy = (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * cfg.initial_pos_range

        # Heading: uniform over (−π, π] or as configured
        psi = (
            (torch.rand(n, device=self.device) - 0.5)
            * 2.0
            * cfg.initial_heading_range
        )

        # Velocities: zero for V1 (vessel starts from rest)
        vel = torch.zeros(n, 3, device=self.device)
        if cfg.initial_vel_range > 0.0:
            vel[:, 0] = (
                (torch.rand(n, device=self.device) - 0.5)
                * 2.0
                * cfg.initial_vel_range
            )

        return torch.cat(
            [xy, psi.unsqueeze(-1), vel], dim=-1
        )  # (n, 6)

    # ------------------------------------------------------------------
    # Physical parameter randomization
    # ------------------------------------------------------------------

    def randomize_hydrodynamics(
        self,
        env_ids: Tensor,
        hydro_model,  # HydrodynamicsModel — typed as Any to avoid circular import
    ) -> None:
        """Apply noise to hydrodynamic parameters for specified environments."""
        if not self.dr_cfg.enabled:
            return
        cfg = self.dr_cfg
        hydro_model.randomize(
            env_ids,
            mass_noise_frac=cfg.mass_noise_frac,
            drag_noise_frac=cfg.drag_noise_frac,
            inertia_noise_frac=cfg.inertia_noise_frac,
        )

    def randomize_thrusters(
        self,
        env_ids: Tensor,
        thrust_model,  # DifferentialThrustModel
    ) -> None:
        """Apply efficiency noise to thrusters for specified environments."""
        if not self.dr_cfg.enabled:
            return
        thrust_model.randomize_efficiency(
            env_ids,
            efficiency_range=self.dr_cfg.thrust_efficiency_range,
        )

    # ------------------------------------------------------------------
    # Disturbance seeding
    # ------------------------------------------------------------------

    def seed_disturbances(
        self,
        env_ids: Tensor,
        wind_model,     # WindDisturbance
        current_model,  # CurrentDisturbance
        wave_model,     # WaveDisturbance
    ) -> None:
        """Reset and re-randomise all disturbance models for given envs."""
        wind_model.reset(env_ids)
        current_model.reset(env_ids)
        wave_model.reset(env_ids)

    # ------------------------------------------------------------------
    # Observation noise
    # ------------------------------------------------------------------

    def add_observation_noise(
        self,
        obs: Tensor,
        pos_noise_std: float,
        vel_noise_std: float,
        heading_noise_std: float,
        yaw_rate_noise_std: float,
    ) -> Tensor:
        """Add Gaussian sensor noise to observation tensor.

        Observation layout assumed: [x, y, psi, u, v, r, ...]

        Args:
            obs: (num_envs, obs_dim)

        Returns:
            noisy_obs: (num_envs, obs_dim)
        """
        noisy = obs.clone()

        # Position (indices 0, 1)
        noisy[:, 0] += pos_noise_std * torch.randn_like(obs[:, 0])
        noisy[:, 1] += pos_noise_std * torch.randn_like(obs[:, 1])

        # Heading (index 2)
        noisy[:, 2] += heading_noise_std * torch.randn_like(obs[:, 2])

        # Velocities (indices 3, 4)
        noisy[:, 3] += vel_noise_std * torch.randn_like(obs[:, 3])
        noisy[:, 4] += vel_noise_std * torch.randn_like(obs[:, 4])

        # Yaw rate (index 5)
        noisy[:, 5] += yaw_rate_noise_std * torch.randn_like(obs[:, 5])

        return noisy
