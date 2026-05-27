"""
Abstract base class for environmental disturbance models.

Each disturbance module is responsible for:
    1. Maintaining internal state (e.g., current direction, wave phase)
    2. Exposing a `compute()` method that returns body-frame wrench
    3. Exposing `step()` to evolve internal state each physics tick
    4. Exposing `reset()` to re-initialise per-episode state
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseDisturbance(ABC):
    """Interface for all disturbance models.

    Disturbances return a body-frame wrench tensor of shape (num_envs, 3)
    representing [X_env, Y_env, N_env] — force in surge, sway, yaw moment.
    """

    def __init__(self, num_envs: int, device: str = "cuda") -> None:
        self.num_envs = num_envs
        self.device = device

    @abstractmethod
    def reset(self, env_ids: Tensor) -> None:
        """Re-initialise disturbance state for specified environments.

        Args:
            env_ids: 1D tensor of environment indices.
        """

    @abstractmethod
    def step(self, dt: float) -> None:
        """Evolve internal disturbance state by dt seconds.

        Called once per physics timestep before `compute()`.

        Args:
            dt: physics timestep (s).
        """

    @abstractmethod
    def compute(self, vessel_state: Tensor) -> Tensor:
        """Return current disturbance wrench.

        Args:
            vessel_state: (num_envs, 6) vessel state [x,y,psi,u,v,r]

        Returns:
            tau_env: (num_envs, 3) body-frame wrench [X, Y, N]
        """

    def _zeros(self) -> Tensor:
        """Convenience — return zero wrench for all envs."""
        return torch.zeros(self.num_envs, 3, device=self.device)
