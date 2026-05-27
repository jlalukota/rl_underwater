"""
3-DOF hydrodynamic force model for a surface vessel.

Implements the Fossen (2011) maneuvering model:

    M * nu_dot + C(nu) * nu + D(nu) * nu = tau + tau_env

Where:
    nu  = [u, v, r]^T  (surge, sway, yaw rate in body frame)
    M   = rigid-body + added-mass inertia matrix
    C   = Coriolis-centripetal matrix
    D   = hydrodynamic damping matrix (linear + quadratic)

All tensors are batched: shape (num_envs, ...) for GPU parallelism.

Reference:
    Fossen, T. I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control.
    Wiley-IEEE Press.
"""

from __future__ import annotations

import torch
from torch import Tensor

from configs.blueboat_cfg import BlueboatDynamicsConfig


class HydrodynamicsModel:
    """Batched 3-DOF hydrodynamic model.

    All operations are pure PyTorch — no Python loops over environments.
    """

    def __init__(
        self,
        cfg: BlueboatDynamicsConfig,
        num_envs: int,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Pre-compute scalar effective inertias
        # These may be randomized per env in domain randomization
        self.m11 = torch.full((num_envs,), cfg.m11, device=device)  # (N,)
        self.m22 = torch.full((num_envs,), cfg.m22, device=device)  # (N,)
        self.m33 = torch.full((num_envs,), cfg.m33, device=device)  # (N,)

        # Linear damping coefficients
        self.Xu = torch.full((num_envs,), cfg.Xu, device=device)
        self.Yv = torch.full((num_envs,), cfg.Yv, device=device)
        self.Nr = torch.full((num_envs,), cfg.Nr, device=device)

        # Quadratic damping coefficients
        self.Xuu = torch.full((num_envs,), cfg.Xuu, device=device)
        self.Yvv = torch.full((num_envs,), cfg.Yvv, device=device)
        self.Nrr = torch.full((num_envs,), cfg.Nrr, device=device)

    # ------------------------------------------------------------------
    # Mass matrix (diagonal for symmetric hull)
    # ------------------------------------------------------------------

    def inv_mass_matrix(self) -> Tensor:
        """Return per-env inverse diagonal of M.  Shape: (num_envs, 3)."""
        return torch.stack(
            [1.0 / self.m11, 1.0 / self.m22, 1.0 / self.m33], dim=-1
        )

    # ------------------------------------------------------------------
    # Coriolis-centripetal forces  C(nu) * nu
    # ------------------------------------------------------------------

    def coriolis_forces(self, nu: Tensor) -> Tensor:
        """Compute Coriolis-centripetal force vector C(nu)*nu.

        For a diagonal mass matrix (symmetric hull, x_g = 0):

            C(nu) = [  0,      0,    -m22*v ]
                    [  0,      0,     m11*u ]
                    [ m22*v, -m11*u,    0   ]

        Args:
            nu: body velocities (num_envs, 3) = [u, v, r]

        Returns:
            Tensor (num_envs, 3) — Coriolis forces in [surge, sway, yaw]
        """
        u = nu[:, 0]
        v = nu[:, 1]
        r = nu[:, 2]

        f_surge = -self.m22 * v * r
        f_sway = self.m11 * u * r
        f_yaw = (self.m22 - self.m11) * u * v

        return torch.stack([f_surge, f_sway, f_yaw], dim=-1)

    # ------------------------------------------------------------------
    # Hydrodynamic damping forces  D(nu) * nu
    # ------------------------------------------------------------------

    def damping_forces(self, nu: Tensor) -> Tensor:
        """Compute hydrodynamic damping force vector (linear + quadratic).

        Simplified decoupled form:
            X_damp = Xu*u + Xuu*|u|*u
            Y_damp = Yv*v + Yvv*|v|*v
            N_damp = Nr*r + Nrr*|r|*r

        All coefficients are negative, so these forces oppose motion.

        Args:
            nu: (num_envs, 3) body velocities [u, v, r]

        Returns:
            Tensor (num_envs, 3) damping forces
        """
        u = nu[:, 0]
        v = nu[:, 1]
        r = nu[:, 2]

        f_surge = self.Xu * u + self.Xuu * u.abs() * u
        f_sway = self.Yv * v + self.Yvv * v.abs() * v
        f_yaw = self.Nr * r + self.Nrr * r.abs() * r

        return torch.stack([f_surge, f_sway, f_yaw], dim=-1)

    # ------------------------------------------------------------------
    # Net body-frame acceleration
    # ------------------------------------------------------------------

    def compute_nu_dot(
        self,
        nu: Tensor,
        tau: Tensor,
        tau_env: Tensor,
        nu_current: Tensor | None = None,
    ) -> Tensor:
        """Compute body-frame velocity derivatives.

        Rearranged Fossen maneuvering equation:

            nu_dot = M^{-1} * (tau + tau_env - C(nu)*nu + f_damp(nu_rel))

        Coriolis uses the absolute body velocity nu (kinetic energy basis).
        Damping uses the *water-relative* velocity nu_rel = nu - nu_current,
        because hydrodynamic drag depends on motion relative to the fluid.

        When nu_current is None the formulation reduces to the single-fluid case.

        Sign audit for pure surge (u > 0, no control, no current):
            f_cor[surge]  = -m22 * 0 * 0 = 0
            f_damp[surge] = Xu*u + Xuu*|u|*u  < 0   (Xu < 0, Xuu < 0)
            net[surge]    = 0 + 0 - 0 + f_damp < 0  → u_dot < 0  ✓ (deceleration)

        Args:
            nu:         (num_envs, 3) absolute body velocities [u, v, r]
            tau:        (num_envs, 3) control forces [X, Y, N]
            tau_env:    (num_envs, 3) non-current environmental wrench
            nu_current: (num_envs, 3) optional current velocity in body frame

        Returns:
            nu_dot: (num_envs, 3) body-frame accelerations
        """
        f_cor = self.coriolis_forces(nu)

        nu_rel = nu if nu_current is None else (nu - nu_current)
        f_damp = self.damping_forces(nu_rel)

        net = tau + tau_env - f_cor + f_damp
        inv_m = self.inv_mass_matrix()
        return net * inv_m

    # ------------------------------------------------------------------
    # Domain randomization support
    # ------------------------------------------------------------------

    def randomize(
        self,
        env_ids: Tensor,
        mass_noise_frac: float,
        drag_noise_frac: float,
        inertia_noise_frac: float,
    ) -> None:
        """Apply multiplicative noise to per-env hydrodynamic parameters.

        Args:
            env_ids:          Indices of environments to randomize.
            mass_noise_frac:  Fractional std for m11/m22 noise.
            drag_noise_frac:  Fractional std for Xu/Yv/Nr/Xuu/Yvv/Nrr noise.
            inertia_noise_frac: Fractional std for m33 noise.
        """
        n = env_ids.numel()
        cfg = self.cfg

        def _perturb(nominal: float, frac: float) -> Tensor:
            noise = 1.0 + frac * torch.randn(n, device=self.device)
            return nominal * noise

        self.m11[env_ids] = _perturb(cfg.m11, mass_noise_frac)
        self.m22[env_ids] = _perturb(cfg.m22, mass_noise_frac)
        self.m33[env_ids] = _perturb(cfg.m33, inertia_noise_frac)

        self.Xu[env_ids] = _perturb(cfg.Xu, drag_noise_frac)
        self.Yv[env_ids] = _perturb(cfg.Yv, drag_noise_frac)
        self.Nr[env_ids] = _perturb(cfg.Nr, drag_noise_frac)
        self.Xuu[env_ids] = _perturb(cfg.Xuu, drag_noise_frac)
        self.Yvv[env_ids] = _perturb(cfg.Yvv, drag_noise_frac)
        self.Nrr[env_ids] = _perturb(cfg.Nrr, drag_noise_frac)

        # Clamp to physically meaningful range (must stay negative)
        self.m11[env_ids].clamp_(min=self.cfg.mass * 0.5)
        self.m22[env_ids].clamp_(min=self.cfg.mass * 0.5)
        self.m33[env_ids].clamp_(min=self.cfg.inertia_z * 0.5)
