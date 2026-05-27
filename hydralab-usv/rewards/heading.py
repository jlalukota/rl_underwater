"""
Heading alignment reward.

Encourages the vessel to face toward the goal (or a desired heading).

R_heading = exp(-e_psi² / (2 * σ_psi²))

where e_psi = wrap(ψ_desired − ψ_current) ∈ (−π, π].

Wrapping is critical: without it gradients become discontinuous at ±π.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rewards.base import BaseReward


def _wrap_angle(angle: Tensor) -> Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


class HeadingReward(BaseReward):
    """Gaussian heading-alignment reward."""

    def __init__(
        self,
        weight: float,
        sigma: float = 0.5,
        device: str = "cuda",
    ) -> None:
        super().__init__(weight, device)
        self.sigma = sigma

    def compute(
        self,
        psi: Tensor,
        goal_heading: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute heading reward.

        Args:
            psi:          (num_envs,) current heading (rad)
            goal_heading: (num_envs,) desired heading (rad)

        Returns:
            reward: (num_envs,)
        """
        error = _wrap_angle(goal_heading - psi)
        return torch.exp(-(error ** 2) / (2.0 * self.sigma ** 2))

    @staticmethod
    def goal_heading_from_pos(
        pos: Tensor,
        goal_pos: Tensor,
    ) -> Tensor:
        """Compute the desired heading pointing from pos toward goal_pos.

        Args:
            pos:      (num_envs, 2)
            goal_pos: (num_envs, 2)

        Returns:
            heading: (num_envs,) in radians
        """
        delta = goal_pos - pos  # (N, 2)
        return torch.atan2(delta[:, 1], delta[:, 0])
