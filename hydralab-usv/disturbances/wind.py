"""
Wind disturbance model.

Models a slowly-varying wind field using a first-order Gauss-Markov process
for both speed and direction.  The wind force on the vessel hull is modelled
as a quadratic drag on the above-waterline projected area:

    F_wind = 0.5 * rho_air * Cw * A_proj * V_rel²

where V_rel is the relative wind velocity in body frame.  For V1 a simplified
lumped coefficient approach is used:

    [X_wind, Y_wind] = Cw * V_rel * |V_rel|   (signed quadratic)
    N_wind           ≈ 0   (symmetric hull)

The wind vector is expressed in the NED world frame; it is transformed to
the vessel body frame each timestep using the current heading ψ.

TODO: Replace lumped Cw with direction-dependent coefficient table (for
      realistic ASV superstructure aerodynamics).
"""

from __future__ import annotations

import torch
from torch import Tensor

from configs.blueboat_cfg import DisturbanceConfig
from disturbances.base import BaseDisturbance


class WindDisturbance(BaseDisturbance):
    """Slowly-varying wind disturbance with Gauss-Markov process."""

    def __init__(
        self,
        cfg: DisturbanceConfig,
        num_envs: int,
        device: str = "cuda",
    ) -> None:
        super().__init__(num_envs, device)
        self.cfg = cfg

        # Internal state: wind speed and direction per environment
        self._speed = torch.zeros(num_envs, device=device)      # m/s
        self._direction = torch.zeros(num_envs, device=device)  # rad (world frame)

        # Time counter for periodic wind updates
        self._time_since_update = torch.zeros(num_envs, device=device)

        # Lumped wind drag coefficient (N/(m/s)²)
        self._Cw = cfg.wind_drag_coeff

        # Initialise with random wind
        all_ids = torch.arange(num_envs, device=device)
        self.reset(all_ids)

    # ------------------------------------------------------------------
    # BaseDisturbance interface
    # ------------------------------------------------------------------

    def reset(self, env_ids: Tensor) -> None:
        """Randomise wind state for specified environments."""
        n = env_ids.numel()
        if n == 0:
            return
        cfg = self.cfg
        self._speed[env_ids] = (
            cfg.wind_speed_mean
            + cfg.wind_speed_std * torch.randn(n, device=self.device)
        ).clamp(min=0.0)
        self._direction[env_ids] = torch.rand(n, device=self.device) * 2.0 * torch.pi
        self._time_since_update[env_ids] = 0.0

    def step(self, dt: float) -> None:
        """Update wind state every `wind_update_period` seconds."""
        self._time_since_update += dt
        mask = self._time_since_update >= self.cfg.wind_update_period
        if mask.any():
            update_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            self._update_wind(update_ids)
            self._time_since_update[update_ids] = 0.0

    def compute(self, vessel_state: Tensor) -> Tensor:
        """Compute wind force in vessel body frame.

        Args:
            vessel_state: (num_envs, 6)

        Returns:
            tau_wind: (num_envs, 3)
        """
        if not self.cfg.wind_enabled:
            return self._zeros()

        psi = vessel_state[:, 2]

        # World-frame wind vector
        Vw_x = self._speed * torch.cos(self._direction)  # (N,)
        Vw_y = self._speed * torch.sin(self._direction)

        # Vessel velocity in world frame
        u = vessel_state[:, 3]
        v = vessel_state[:, 4]
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)
        Vv_x = u * cos_psi - v * sin_psi
        Vv_y = u * sin_psi + v * cos_psi

        # Relative wind in world frame
        Vrel_x = Vw_x - Vv_x
        Vrel_y = Vw_y - Vv_y

        # Rotate relative wind to body frame
        Vrel_body_u = Vrel_x * cos_psi + Vrel_y * sin_psi    # surge component
        Vrel_body_v = -Vrel_x * sin_psi + Vrel_y * cos_psi   # sway component

        # Signed quadratic drag
        X_wind = self._Cw * Vrel_body_u * Vrel_body_u.abs()
        Y_wind = self._Cw * Vrel_body_v * Vrel_body_v.abs()
        N_wind = torch.zeros_like(X_wind)

        return torch.stack([X_wind, Y_wind, N_wind], dim=-1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_wind(self, env_ids: Tensor) -> None:
        """Apply Gauss-Markov perturbation to wind for given envs."""
        n = env_ids.numel()
        cfg = self.cfg
        new_speed = (
            cfg.wind_speed_mean
            + cfg.wind_speed_std * torch.randn(n, device=self.device)
        ).clamp(min=0.0)
        direction_delta = 0.3 * torch.randn(n, device=self.device)  # rad

        self._speed[env_ids] = new_speed
        self._direction[env_ids] = (
            self._direction[env_ids] + direction_delta
        ) % (2.0 * torch.pi)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def wind_speed(self) -> Tensor:
        return self._speed

    @property
    def wind_direction(self) -> Tensor:
        return self._direction
