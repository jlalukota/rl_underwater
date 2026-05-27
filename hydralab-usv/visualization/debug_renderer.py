"""
Debug renderer for HydraLab-USV.

Provides 2D matplotlib-based visualisation for debugging, trajectory
inspection, and reward signal analysis.  NOT intended for cinematic
rendering or photorealistic ocean display.

Features:
    - Top-down 2D trajectory traces
    - Thrust vector overlays
    - Heading arrows
    - Disturbance force arrows
    - Goal / waypoint markers
    - Reward signal time-series

Usage:
    renderer = DebugRenderer(env, max_envs_to_show=4)
    for step in range(200):
        obs, rew, *_ = env.step(action)
        renderer.update(env)
    renderer.show()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


class DebugRenderer:
    """Lightweight 2D debug visualiser."""

    def __init__(
        self,
        num_envs_to_show: int = 4,
        history_len: int = 200,
        figsize: tuple[int, int] = (14, 10),
    ) -> None:
        if not _MPL_AVAILABLE:
            raise ImportError("matplotlib is required for DebugRenderer.")

        self.num_show = num_envs_to_show
        self.history_len = history_len

        # Position history buffers  (max_show, history_len, 2)
        self._pos_history: list[list[tuple[float, float]]] = [
            [] for _ in range(num_envs_to_show)
        ]
        self._reward_history: list[list[float]] = [
            [] for _ in range(num_envs_to_show)
        ]

        # Setup figure
        self._fig, self._axes = plt.subplots(
            2, num_envs_to_show,
            figsize=figsize,
            squeeze=False,
        )
        self._fig.suptitle("HydraLab-USV Debug View", fontsize=12)

        for i in range(num_envs_to_show):
            ax = self._axes[0, i]
            ax.set_title(f"Env {i}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

            ax2 = self._axes[1, i]
            ax2.set_title(f"Reward env {i}")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("R")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        env,  # BlueboatEnv
        rewards: Optional[Tensor] = None,
    ) -> None:
        """Record current environment state for visualisation.

        Args:
            env:     BlueboatEnv instance.
            rewards: (num_envs,) optional reward tensor this step.
        """
        state = env.body.state  # (N, 6)
        n_show = min(self.num_show, env.num_envs)

        for i in range(n_show):
            x = float(state[i, 0].cpu())
            y = float(state[i, 1].cpu())
            self._pos_history[i].append((x, y))
            if len(self._pos_history[i]) > self.history_len:
                self._pos_history[i].pop(0)

            if rewards is not None:
                r = float(rewards[i].cpu())
                self._reward_history[i].append(r)
                if len(self._reward_history[i]) > self.history_len:
                    self._reward_history[i].pop(0)

    def render(self, env) -> None:
        """Re-draw the debug figure with current state.

        Args:
            env: BlueboatEnv instance (reads current state from it).
        """
        state = env.body.state.cpu()
        goal = env.task.target_pos.cpu()
        n_show = min(self.num_show, env.num_envs)

        for i in range(n_show):
            ax = self._axes[0, i]
            ax.cla()
            ax.set_title(f"Env {i}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Trajectory trace
            if self._pos_history[i]:
                xs, ys = zip(*self._pos_history[i])
                ax.plot(xs, ys, "b-", linewidth=0.8, alpha=0.7, label="trajectory")
                ax.plot(xs[-1], ys[-1], "bo", markersize=6)

            # Goal marker
            gx, gy = float(goal[i, 0]), float(goal[i, 1])
            ax.plot(gx, gy, "r*", markersize=12, label="goal")

            # Heading arrow
            x = float(state[i, 0])
            y = float(state[i, 1])
            psi = float(state[i, 2])
            arrow_len = 2.0
            ax.annotate(
                "",
                xy=(x + arrow_len * np.cos(psi), y + arrow_len * np.sin(psi)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
            )

            # Workspace boundary
            theta = np.linspace(0, 2 * np.pi, 100)
            R = env.cfg.env.workspace_radius
            ax.plot(R * np.cos(theta), R * np.sin(theta), "k--", alpha=0.3)

            ax.legend(fontsize=7, loc="upper right")

            # Reward plot
            ax2 = self._axes[1, i]
            ax2.cla()
            ax2.set_title(f"Reward env {i}")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("R")
            ax2.grid(True, alpha=0.3)
            if self._reward_history[i]:
                ax2.plot(self._reward_history[i], "g-", linewidth=0.8)

        plt.tight_layout()
        plt.pause(0.001)

    def show(self) -> None:
        """Display the final figure."""
        plt.show()

    def save(self, path: str) -> None:
        """Save figure to file."""
        self._fig.savefig(path, dpi=150, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Static trajectory plotter (standalone, no live env needed)
    # ------------------------------------------------------------------

    @staticmethod
    def plot_trajectory(
        positions: np.ndarray,
        goals: Optional[np.ndarray] = None,
        title: str = "Vessel Trajectory",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a single trajectory from a numpy position array.

        Args:
            positions:  (T, 2) array of world positions
            goals:      (T, 2) optional goal positions
            title:      figure title
            save_path:  if provided, saves figure here
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(positions[:, 0], positions[:, 1], "b-", label="vessel")
        ax.plot(positions[0, 0], positions[0, 1], "go", markersize=10, label="start")
        ax.plot(positions[-1, 0], positions[-1, 1], "ro", markersize=10, label="end")
        if goals is not None:
            ax.plot(goals[:, 0], goals[:, 1], "r--", alpha=0.5, label="reference")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
