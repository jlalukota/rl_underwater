"""
Boundary penalty reward.

Applies a soft growing penalty as the vessel approaches the workspace edge,
and a hard constant penalty once it crosses the boundary.

Penalty shape:
    - Zero inside  (workspace_radius - margin)
    - Linearly increasing between (workspace_radius - margin) and workspace_radius
    - Constant = hard_penalty beyond workspace_radius

This two-zone design:
    1. Provides a smooth gradient signal before the reset occurs
    2. Makes boundary avoidance learnable without discontinuous reward jumps
"""

from __future__ import annotations

import torch
from torch import Tensor

from rewards.base import BaseReward


class BoundaryReward(BaseReward):
    """Soft + hard boundary violation penalty."""

    def __init__(
        self,
        weight: float,
        workspace_radius: float,
        margin: float = 5.0,
        hard_penalty: float = 20.0,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            weight:           Penalty weight (positive; caller subtracts).
            workspace_radius: Hard boundary radius (m).
            margin:           Soft-boundary band width inside the hard limit (m).
            hard_penalty:     Constant penalty applied beyond workspace_radius.
        """
        super().__init__(weight, device)
        self.workspace_radius = workspace_radius
        self.margin = margin
        self.hard_penalty = hard_penalty
        self._soft_start = workspace_radius - margin

    def compute(self, pos: Tensor, **kwargs) -> Tensor:
        """Compute boundary penalty.

        Args:
            pos: (num_envs, 2) world position [x, y]

        Returns:
            penalty: (num_envs,) positive scalar (caller subtracts)
        """
        dist = pos.norm(dim=-1)

        # Soft linear ramp in the margin zone
        soft = (dist - self._soft_start).clamp(min=0.0)

        # Hard constant once past the boundary
        hard = self.hard_penalty * (dist > self.workspace_radius).float()

        return soft + hard

    def is_violated(self, pos: Tensor) -> Tensor:
        """(num_envs,) bool — True if vessel is outside workspace."""
        return pos.norm(dim=-1) > self.workspace_radius

    def distance_to_boundary(self, pos: Tensor) -> Tensor:
        """(num_envs,) signed distance to workspace edge (negative = inside)."""
        return pos.norm(dim=-1) - self.workspace_radius
