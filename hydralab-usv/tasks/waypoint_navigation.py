"""
Waypoint navigation task.

A sequence of waypoints is sampled at episode start.  The vessel must
reach each waypoint within `goal_radius` to advance to the next one.
The episode succeeds when all waypoints are visited or when the
step budget is exhausted.

Key behaviours:
    - Sparse bonus on waypoint capture
    - Dense tracking reward toward current waypoint
    - Per-environment independent waypoint sequences
"""

from __future__ import annotations

import torch
from torch import Tensor

from configs.blueboat_cfg import EnvironmentConfig


class WaypointNavigationTask:
    """Sequential waypoint navigation with per-env waypoint queues."""

    def __init__(
        self,
        env_cfg: EnvironmentConfig,
        num_envs: int,
        device: str = "cuda",
        num_waypoints: int = 5,
        waypoint_range: float = 20.0,
    ) -> None:
        self.env_cfg = env_cfg
        self.num_envs = num_envs
        self.device = device
        self.num_waypoints = num_waypoints
        self.waypoint_range = waypoint_range

        # Waypoint storage: (num_envs, num_waypoints, 2)
        self._waypoints = torch.zeros(
            num_envs, num_waypoints, 2, device=device
        )
        # Current waypoint index per env
        self._wp_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

        all_ids = torch.arange(num_envs, device=device)
        self.reset(all_ids)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: Tensor) -> None:
        """Sample fresh waypoint sequences for specified environments."""
        n = env_ids.numel()
        if n == 0:
            return
        r = self.waypoint_range
        self._waypoints[env_ids] = (
            torch.rand(n, self.num_waypoints, 2, device=self.device) - 0.5
        ) * 2.0 * r
        self._wp_idx[env_ids] = 0

    # ------------------------------------------------------------------
    # Target accessors
    # ------------------------------------------------------------------

    @property
    def current_waypoint(self) -> Tensor:
        """(num_envs, 2) current active waypoint position."""
        idx = self._wp_idx.clamp(max=self.num_waypoints - 1)
        return self._waypoints[
            torch.arange(self.num_envs, device=self.device), idx
        ]

    def get_waypoint_progress(self) -> Tensor:
        """(num_envs,) fraction of waypoints completed ∈ [0, 1]."""
        return self._wp_idx.float() / self.num_waypoints

    # ------------------------------------------------------------------
    # Check capture and advance
    # ------------------------------------------------------------------

    def check_and_advance(self, pos: Tensor) -> Tensor:
        """Check if any environments reached their current waypoint.

        Advances the waypoint index for those environments.

        Args:
            pos: (num_envs, 2) current vessel positions

        Returns:
            captured: (num_envs,) bool tensor — True if waypoint just captured
        """
        dist = ((pos - self.current_waypoint) ** 2).sum(dim=-1).sqrt()
        captured = (dist < self.env_cfg.goal_radius) & (
            self._wp_idx < self.num_waypoints
        )
        self._wp_idx[captured] = (self._wp_idx[captured] + 1).clamp(
            max=self.num_waypoints
        )
        return captured

    def all_complete(self) -> Tensor:
        """(num_envs,) bool — True if all waypoints visited."""
        return self._wp_idx >= self.num_waypoints
