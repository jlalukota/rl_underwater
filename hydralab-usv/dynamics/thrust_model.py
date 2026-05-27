"""
Differential thrust model for a two-thruster catamaran.

Maps normalized thruster commands → body-frame forces and moments.

Action space:  Box(low=-1, high=1, shape=(2,))
    action[0] = left  (port)  thruster command
    action[1] = right (stbd)  thruster command

Body-frame wrench produced:
    X = F_L + F_R           (surge force)
    Y = 0                   (no lateral thruster)
    N = (F_R - F_L) * B/2   (yaw moment, B = thruster separation)

TODO: Add thruster saturation model (current-limited at low RPM).
TODO: Replace quadratic mapping with polynomial fit from T200 data sheet.
"""

from __future__ import annotations

import torch
from torch import Tensor

from configs.blueboat_cfg import ThrustConfig, BlueboatDynamicsConfig


class DifferentialThrustModel:
    """Converts normalized commands to body-frame control wrench.

    Supports per-env thruster efficiency randomization.
    """

    def __init__(
        self,
        thrust_cfg: ThrustConfig,
        dynamics_cfg: BlueboatDynamicsConfig,
        num_envs: int,
        device: str = "cuda",
    ) -> None:
        self.thrust_cfg = thrust_cfg
        self.num_envs = num_envs
        self.device = device

        self.beam = dynamics_cfg.beam

        # Per-env thruster efficiency scalars (1 = nominal)
        self.efficiency = torch.ones(num_envs, 2, device=device)  # (N, 2)

    # ------------------------------------------------------------------
    # Core mapping
    # ------------------------------------------------------------------

    def normalized_to_force(self, cmd: Tensor) -> Tensor:
        """Map a normalized command ∈ [-1, 1] to thruster force (N).

        Uses an asymmetric linear map matching T200 forward/reverse asymmetry:
            cmd > 0: F = cmd * max_thrust_fwd
            cmd < 0: F = |cmd| * max_thrust_rev  (sign preserved)

        Args:
            cmd: (num_envs, 2) normalized commands

        Returns:
            forces: (num_envs, 2) per-thruster forces in Newtons
        """
        cfg = self.thrust_cfg

        # Apply deadband
        cmd = cmd.clone()
        cmd[cmd.abs() < cfg.deadband] = 0.0

        positive_part = cmd.clamp(min=0.0) * cfg.max_thrust_fwd
        negative_part = cmd.clamp(max=0.0) * cfg.max_thrust_rev

        forces = positive_part + negative_part  # (N, 2)

        # Apply per-env efficiency
        forces = forces * self.efficiency

        return forces

    def forces_to_wrench(self, forces: Tensor) -> Tensor:
        """Convert per-thruster forces to body-frame control wrench.

        Args:
            forces: (num_envs, 2) [F_L, F_R] in Newtons

        Returns:
            tau: (num_envs, 3) [X, Y, N] control wrench
        """
        F_L = forces[:, 0]
        F_R = forces[:, 1]

        X = F_L + F_R                          # surge force (N)
        Y = torch.zeros_like(X)                # no lateral thruster
        N = (F_R - F_L) * (self.beam / 2.0)   # yaw moment (N·m)

        return torch.stack([X, Y, N], dim=-1)

    def compute_tau(self, actions: Tensor) -> Tensor:
        """End-to-end: normalized actions → body wrench [X, Y, N].

        Args:
            actions: (num_envs, 2) in [-1, 1]

        Returns:
            tau: (num_envs, 3) control wrench
        """
        forces = self.normalized_to_force(actions)
        return self.forces_to_wrench(forces)

    # ------------------------------------------------------------------
    # First-order actuator lag (optional, for higher fidelity)
    # ------------------------------------------------------------------

    def apply_lag(
        self,
        current_forces: Tensor,
        commanded_forces: Tensor,
        dt: float,
    ) -> Tensor:
        """First-order low-pass filter on thruster forces.

        F_new = F_cur + (dt / tc) * (F_cmd - F_cur)

        Args:
            current_forces:  (num_envs, 2) current actual forces
            commanded_forces:(num_envs, 2) commanded forces
            dt:              timestep (s)

        Returns:
            updated_forces: (num_envs, 2)
        """
        tc = self.thrust_cfg.time_constant
        alpha = dt / tc
        return current_forces + alpha * (commanded_forces - current_forces)

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------

    def randomize_efficiency(
        self,
        env_ids: Tensor,
        efficiency_range: tuple[float, float],
    ) -> None:
        """Sample per-thruster efficiency for specified environments.

        Args:
            env_ids:           Indices of environments to randomize.
            efficiency_range:  (min, max) uniform efficiency factor.
        """
        lo, hi = efficiency_range
        n = env_ids.numel()
        self.efficiency[env_ids] = (
            torch.rand(n, 2, device=self.device) * (hi - lo) + lo
        )
