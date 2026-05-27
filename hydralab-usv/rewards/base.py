"""
Abstract base class for reward components.

Each reward module computes a scalar reward tensor of shape (num_envs,)
from the current environment state.  Modules are composed in the environment
via a weighted sum.

Design principle: reward components must be STATELESS with respect to the
environment — they only read tensors passed as arguments, never store state.
This makes them trivially unit-testable and easily composable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseReward(ABC):
    """Interface for all reward components."""

    def __init__(self, weight: float, device: str = "cuda") -> None:
        self.weight = weight
        self.device = device

    @abstractmethod
    def compute(self, **kwargs) -> Tensor:
        """Compute per-environment reward scalar.

        Returns:
            reward: (num_envs,) unweighted reward values
        """

    def weighted(self, **kwargs) -> Tensor:
        """Return weight * compute(...)."""
        return self.weight * self.compute(**kwargs)
