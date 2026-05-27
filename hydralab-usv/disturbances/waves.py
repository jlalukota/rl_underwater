"""
Stochastic wave force injection.

Wave loads on a small surface vessel are complex (radiation, diffraction,
Froude-Krylov, etc.).  For V1 we use a simple coloured-noise injection
model: a first-order Gauss-Markov process with specified bandwidth.

This produces forces and moments that:
    - Have a natural frequency consistent with short ocean waves
    - Are spatially independent between environments
    - Are zero-mean over long episodes (no steady drift)
    - Scale with a configurable standard deviation

The Ornstein-Uhlenbeck update is:
    F(t+dt) = F(t) * exp(-beta*dt) + sigma * sqrt(1-exp(-2*beta*dt)) * N(0,1)

where beta = 2*pi*bandwidth is the correlation frequency.

TODO: Replace with spectral wave model (JONSWAP or Pierson-Moskowitz)
      for more realistic wave-period statistics.
TODO: Incorporate heading-dependent wave encounter frequency.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from configs.blueboat_cfg import DisturbanceConfig
from disturbances.base import BaseDisturbance


class WaveDisturbance(BaseDisturbance):
    """Coloured-noise (Ornstein-Uhlenbeck) wave force model."""

    def __init__(
        self,
        cfg: DisturbanceConfig,
        num_envs: int,
        device: str = "cuda",
    ) -> None:
        super().__init__(num_envs, device)
        self.cfg = cfg

        # Current force / moment state per environment
        # Layout: [X_wave, Y_wave, N_wave]
        self._forces = torch.zeros(num_envs, 3, device=device)

        # OU process parameters
        self._beta = 2.0 * math.pi * cfg.wave_bandwidth  # rad/s

        # Per-component noise std
        self._sigma = torch.tensor(
            [cfg.wave_force_std, cfg.wave_force_std, cfg.wave_moment_std],
            device=device,
        )  # (3,)

        all_ids = torch.arange(num_envs, device=device)
        self.reset(all_ids)

    # ------------------------------------------------------------------
    # BaseDisturbance interface
    # ------------------------------------------------------------------

    def reset(self, env_ids: Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._forces[env_ids] = 0.0

    def step(self, dt: float) -> None:
        """Advance OU process by dt.

        F(t+dt) = decay * F(t) + noise_std * randn
        """
        if not self.cfg.wave_enabled:
            return

        decay = math.exp(-self._beta * dt)
        noise_std = self._sigma * math.sqrt(1.0 - decay**2)

        noise = torch.randn_like(self._forces)  # (N, 3)
        noise = noise * noise_std.unsqueeze(0)   # broadcast sigma

        self._forces = decay * self._forces + noise

    def compute(self, vessel_state: Tensor) -> Tensor:
        """Return current wave force tensor.

        Args:
            vessel_state: (num_envs, 6)  (unused, waves are position-independent)

        Returns:
            tau_wave: (num_envs, 3)
        """
        if not self.cfg.wave_enabled:
            return self._zeros()
        return self._forces.clone()
