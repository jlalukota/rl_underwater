"""
Energy consumption penalty.

Penalises high thruster commands to encourage energy-efficient trajectories.
Proxy for battery consumption, modelled as:

    P_energy = mean(a²)

For real BlueBoat, power ≈ I * V; current is roughly proportional to thrust,
so thrust² ∝ power.  This proxy is monotone with actual power draw.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rewards.base import BaseReward


class EnergyReward(BaseReward):
    """Quadratic actuation energy penalty."""

    def compute(
        self,
        action: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute energy penalty.

        Args:
            action: (num_envs, 2) normalized thruster commands in [-1, 1]

        Returns:
            penalty: (num_envs,) positive scalar (caller should subtract)
        """
        return (action ** 2).mean(dim=-1)
