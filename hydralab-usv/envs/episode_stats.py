"""
Per-episode statistics accumulator.

Tracks running metrics across each environment's current episode.
When an episode ends (done=True), the summary is collected and the
accumulators are reset for that environment.

Tracked metrics:
    - cumulative reward
    - step count
    - mean tracking error (RMSE-like)
    - minimum distance to goal achieved
    - boundary violations count

Usage in the environment step loop:
    stats.update(rewards, dist_to_goal, boundary_flags)
    summaries = stats.collect_done(done_mask)  # returns list of dicts for done envs
    stats.reset(done_ids)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class EpisodeStats:
    """Accumulates per-episode metrics across N parallel environments."""

    def __init__(self, num_envs: int, device: str = "cuda") -> None:
        self.num_envs = num_envs
        self.device = device

        self._cum_reward = torch.zeros(num_envs, device=device)
        self._cum_tracking_err = torch.zeros(num_envs, device=device)
        self._min_dist = torch.full((num_envs,), float("inf"), device=device)
        self._step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._boundary_hits = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Completed episode records (appended on each done event)
        self._completed: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update(
        self,
        rewards: Tensor,
        dist_to_goal: Tensor,
        boundary_violated: Tensor | None = None,
    ) -> None:
        """Accumulate one step of statistics.

        Args:
            rewards:           (num_envs,) per-env rewards this step.
            dist_to_goal:      (num_envs,) distance to current goal.
            boundary_violated: (num_envs,) bool — True if OOB this step.
        """
        self._cum_reward += rewards
        self._cum_tracking_err += dist_to_goal
        self._min_dist = torch.minimum(self._min_dist, dist_to_goal)
        self._step_count += 1
        if boundary_violated is not None:
            self._boundary_hits += boundary_violated.long()

    # ------------------------------------------------------------------
    # Episode completion
    # ------------------------------------------------------------------

    def collect_done(self, done_mask: Tensor) -> list[dict[str, Any]]:
        """Collect stats for environments whose episodes just ended.

        Args:
            done_mask: (num_envs,) bool — True for completed environments.

        Returns:
            List of stat dicts, one per done environment.
        """
        summaries: list[dict[str, Any]] = []
        done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)

        for idx in done_ids.tolist():
            steps = int(self._step_count[idx])
            summary = {
                "env_id": idx,
                "episode_return": float(self._cum_reward[idx]),
                "episode_length": steps,
                "mean_tracking_error": float(
                    self._cum_tracking_err[idx] / max(steps, 1)
                ),
                "min_dist_to_goal": float(self._min_dist[idx]),
                "boundary_hits": int(self._boundary_hits[idx]),
            }
            summaries.append(summary)
            self._completed.append(summary)

        return summaries

    def reset(self, env_ids: Tensor) -> None:
        """Reset accumulators for specified environments."""
        self._cum_reward[env_ids] = 0.0
        self._cum_tracking_err[env_ids] = 0.0
        self._min_dist[env_ids] = float("inf")
        self._step_count[env_ids] = 0
        self._boundary_hits[env_ids] = 0

    # ------------------------------------------------------------------
    # Aggregate statistics across completed episodes
    # ------------------------------------------------------------------

    def summary(self, last_n: int = 100) -> dict[str, float]:
        """Return mean stats over the last N completed episodes.

        Args:
            last_n: number of most recent episodes to average.

        Returns:
            dict with mean_return, mean_length, mean_tracking_error,
            mean_min_dist, mean_boundary_hits.
        """
        if not self._completed:
            return {}
        recent = self._completed[-last_n:]
        n = len(recent)
        return {
            "mean_return": sum(e["episode_return"] for e in recent) / n,
            "mean_length": sum(e["episode_length"] for e in recent) / n,
            "mean_tracking_error": sum(e["mean_tracking_error"] for e in recent) / n,
            "mean_min_dist_to_goal": sum(e["min_dist_to_goal"] for e in recent) / n,
            "mean_boundary_hits": sum(e["boundary_hits"] for e in recent) / n,
            "n_episodes": len(self._completed),
        }

    def clear_history(self) -> None:
        """Discard stored episode history (free memory during long runs)."""
        self._completed.clear()

    # ------------------------------------------------------------------
    # Live diagnostics (current in-progress episodes)
    # ------------------------------------------------------------------

    @property
    def current_returns(self) -> Tensor:
        """(num_envs,) cumulative reward so far in current episode."""
        return self._cum_reward

    @property
    def current_lengths(self) -> Tensor:
        """(num_envs,) step count so far in current episode."""
        return self._step_count
