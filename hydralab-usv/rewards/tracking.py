"""
Position tracking reward.

Provides a dense, Gaussian-shaped reward based on the distance between
the vessel's current world position and the current target waypoint.

R_track = exp(-d² / (2 * σ²))

where d is the Euclidean distance to the goal.

This shaping:
    - Returns ≈ 1.0 when exactly on target
    - Returns ≈ 0.6 at d = σ
    - Returns ≈ 0.0 far from target
    - Is always positive (encourages approach without terminal penalty)
    - Gradient is always well-defined (no discontinuity)

σ (sigma) controls how sharply the reward peaks at the goal.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rewards.base import BaseReward


class TrackingReward(BaseReward):
    """Gaussian potential-based position tracking reward."""

    def __init__(
        self,
        weight: float,
        sigma: float = 1.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(weight, device)
        self.sigma = sigma

    def compute(
        self,
        pos: Tensor,
        goal_pos: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute tracking reward.

        Args:
            pos:      (num_envs, 2) current world position [x, y]
            goal_pos: (num_envs, 2) target position

        Returns:
            reward: (num_envs,)
        """
        dist_sq = ((pos - goal_pos) ** 2).sum(dim=-1)  # (N,)
        return torch.exp(-dist_sq / (2.0 * self.sigma ** 2))

    def distance(self, pos: Tensor, goal_pos: Tensor) -> Tensor:
        """Return Euclidean distance to goal (useful for logging)."""
        return ((pos - goal_pos) ** 2).sum(dim=-1).sqrt()
