"""
Station keeping task.

The vessel must remain within a small radius of a fixed hold position
while rejecting external disturbances.  This tests the controller's
ability to produce stable equilibria under persistent perturbation.

Difficulty can be scaled by increasing disturbance magnitude.
"""

from __future__ import annotations

import torch
from torch import Tensor

from configs.blueboat_cfg import EnvironmentConfig


class StationKeepingTask:
    """Fixed-point hold task with disturbance rejection."""

    def __init__(
        self,
        env_cfg: EnvironmentConfig,
        num_envs: int,
        device: str = "cuda",
        hold_radius: float = 1.0,
    ) -> None:
        self.env_cfg = env_cfg
        self.num_envs = num_envs
        self.device = device
        self.hold_radius = hold_radius

        # Hold position per environment
        self._hold_pos = torch.zeros(num_envs, 2, device=device)
        # Desired heading at hold position
        self._hold_heading = torch.zeros(num_envs, device=device)

        all_ids = torch.arange(num_envs, device=device)
        self.reset(all_ids)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: Tensor) -> None:
        """Set hold positions for specified environments."""
        n = env_ids.numel()
        if n == 0:
            return
        # Hold positions scattered within a moderate area
        self._hold_pos[env_ids] = (
            torch.rand(n, 2, device=self.device) - 0.5
        ) * 20.0
        self._hold_heading[env_ids] = (
            torch.rand(n, device=self.device) - 0.5
        ) * 2.0 * torch.pi

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def target_pos(self) -> Tensor:
        """(num_envs, 2) hold position."""
        return self._hold_pos

    @property
    def target_heading(self) -> Tensor:
        """(num_envs,) desired hold heading."""
        return self._hold_heading

    def is_holding(self, pos: Tensor) -> Tensor:
        """(num_envs,) bool — True if vessel is within hold_radius."""
        dist = ((pos - self._hold_pos) ** 2).sum(dim=-1).sqrt()
        return dist < self.hold_radius

    def hold_time_fraction(self, hold_steps: Tensor, total_steps: int) -> Tensor:
        """Compute fraction of episode spent in hold zone.

        Args:
            hold_steps: (num_envs,) cumulative steps inside hold_radius
            total_steps: episode length in steps

        Returns:
            fraction: (num_envs,) ∈ [0, 1]
        """
        return hold_steps.float() / max(total_steps, 1)
