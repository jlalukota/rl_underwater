"""
Surface current disturbance model.

Phase 2 implementation uses the correct relative-velocity formulation:

    nu_rel = nu - nu_current_body

Hydrodynamic damping is computed using nu_rel (water-relative velocity)
rather than absolute body velocity.  This is the standard Fossen treatment.

The `compute()` method returns zeros: current effects are handled by passing
`get_body_frame_velocity()` into `HydrodynamicsModel.compute_nu_dot()`.

World-frame current velocity is modelled as a slowly-varying Gauss-Markov
process; direction drifts with small-angle random walk each update period.
"""

from __future__ import annotations

import torch
from torch import Tensor

from configs.blueboat_cfg import DisturbanceConfig
from disturbances.base import BaseDisturbance


class CurrentDisturbance(BaseDisturbance):
    """Slowly-varying horizontal current disturbance.

    Current effects enter the dynamics via relative velocity — call
    `get_body_frame_velocity(vessel_state)` each physics step and pass the
    result to `RigidBody3DOF.step(... nu_current=...)`.
    """

    def __init__(
        self,
        cfg: DisturbanceConfig,
        num_envs: int,
        device: str = "cuda",
    ) -> None:
        super().__init__(num_envs, device)
        self.cfg = cfg

        self._speed = torch.zeros(num_envs, device=device)
        self._direction = torch.zeros(num_envs, device=device)  # rad, NED world frame
        self._time_since_update = torch.zeros(num_envs, device=device)

        all_ids = torch.arange(num_envs, device=device)
        self.reset(all_ids)

    # ------------------------------------------------------------------
    # BaseDisturbance interface
    # ------------------------------------------------------------------

    def reset(self, env_ids: Tensor) -> None:
        n = env_ids.numel()
        if n == 0:
            return
        cfg = self.cfg
        self._speed[env_ids] = (
            cfg.current_speed_mean
            + cfg.current_speed_std * torch.randn(n, device=self.device)
        ).clamp(min=0.0)
        self._direction[env_ids] = torch.rand(n, device=self.device) * 2.0 * torch.pi
        self._time_since_update[env_ids] = 0.0

    def step(self, dt: float) -> None:
        """Drift current direction/speed via Gauss-Markov update."""
        self._time_since_update += dt
        mask = self._time_since_update >= self.cfg.current_update_period
        if mask.any():
            update_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            self._update_current(update_ids)
            self._time_since_update[update_ids] = 0.0

    def compute(self, vessel_state: Tensor) -> Tensor:
        """Returns zeros — current is handled via relative-velocity damping.

        This method satisfies the BaseDisturbance interface.  Do NOT add
        current forces here; instead use `get_body_frame_velocity()`.

        Args:
            vessel_state: (num_envs, 6)  (unused)

        Returns:
            zeros: (num_envs, 3)
        """
        return self._zeros()

    # ------------------------------------------------------------------
    # Relative-velocity interface  (Phase 2 correct formulation)
    # ------------------------------------------------------------------

    def get_body_frame_velocity(self, vessel_state: Tensor) -> Tensor:
        """Return current velocity expressed in the vessel body frame.

        Rotates the world-frame current vector into body frame using ψ.
        The yaw component is zero (depth-uniform current has no spin).

        Args:
            vessel_state: (num_envs, 6)

        Returns:
            nu_current: (num_envs, 3)  [u_c, v_c, 0]
        """
        if not self.cfg.current_enabled:
            return torch.zeros(self.num_envs, 3, device=self.device)

        psi = vessel_state[:, 2]
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)

        # World-frame current vector
        Vc_x = self._speed * torch.cos(self._direction)
        Vc_y = self._speed * torch.sin(self._direction)

        # Rotate to body frame: R(ψ)^T * V_world
        u_c = Vc_x * cos_psi + Vc_y * sin_psi
        v_c = -Vc_x * sin_psi + Vc_y * cos_psi
        r_c = torch.zeros_like(u_c)

        return torch.stack([u_c, v_c, r_c], dim=-1)

    def get_world_velocity(self) -> Tensor:
        """Return (num_envs, 2) world-frame current [Vx, Vy] for diagnostics."""
        Vc_x = self._speed * torch.cos(self._direction)
        Vc_y = self._speed * torch.sin(self._direction)
        return torch.stack([Vc_x, Vc_y], dim=-1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_current(self, env_ids: Tensor) -> None:
        n = env_ids.numel()
        cfg = self.cfg
        self._speed[env_ids] = (
            cfg.current_speed_mean
            + cfg.current_speed_std * torch.randn(n, device=self.device)
        ).clamp(min=0.0)
        # Direction drifts slowly (random walk with small angular std)
        self._direction[env_ids] = (
            self._direction[env_ids]
            + 0.2 * torch.randn(n, device=self.device)
        ) % (2.0 * torch.pi)

    @property
    def speed(self) -> Tensor:
        return self._speed

    @property
    def direction(self) -> Tensor:
        return self._direction
