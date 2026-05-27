"""
Control smoothness penalty.

Penalises large changes in thruster commands between consecutive timesteps.
This discourages high-frequency oscillation (chattering) which causes
mechanical wear and inefficiency in real hardware.

P_smooth = mean(Δa²)

where Δa = a_t − a_{t-1} is the action delta vector.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rewards.base import BaseReward


class SmoothnessReward(BaseReward):
    """Action-delta smoothness penalty (negative reward contribution)."""

    def __init__(
        self,
        weight: float,
        delta_clip: float = 2.0,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            weight:      Penalty weight (positive; subtracted in reward sum).
            delta_clip:  Clip delta before squaring to prevent extreme gradients.
        """
        super().__init__(weight, device)
        self.delta_clip = delta_clip

    def compute(
        self,
        action: Tensor,
        prev_action: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute smoothness penalty.

        Args:
            action:      (num_envs, 2) current actions
            prev_action: (num_envs, 2) previous actions

        Returns:
            penalty: (num_envs,) positive scalar (caller should subtract)
        """
        delta = (action - prev_action).clamp(-self.delta_clip, self.delta_clip)
        return (delta ** 2).mean(dim=-1)
