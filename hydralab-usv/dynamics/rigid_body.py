"""
3-DOF rigid body kinematics and numerical integrator.

State vector:  q = [x, y, psi, u, v, r]
    x, y   – NED position (m)
    psi    – heading/yaw (rad, 0 = North/+X, positive CW looking down)
    u      – surge velocity (m/s, body +X)
    v      – sway  velocity (m/s, body +Y, positive to starboard)
    r      – yaw rate (rad/s, positive clockwise from above)

Kinematics (body → world):
    ẋ   = u·cos(ψ) − v·sin(ψ)
    ẏ   = u·sin(ψ) + v·cos(ψ)
    ψ̇   = r

Dynamics:
    ν̇ = M⁻¹ · (τ + τ_env − C(ν)·ν − D(ν)·ν)

Integration:
    Default: RK4 (4th-order Runge-Kutta) for accuracy
    Fallback: Euler for speed when dt is small

All operations are batched over num_envs for GPU parallelism.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from configs.blueboat_cfg import BlueboatDynamicsConfig
from dynamics.hydrodynamics import HydrodynamicsModel
from dynamics.thrust_model import DifferentialThrustModel


# ---------------------------------------------------------------------------
# Pure kinematic helpers
# ---------------------------------------------------------------------------

def body_to_world(nu: Tensor, psi: Tensor) -> Tensor:
    """Rotate body-frame velocities to world frame.

    Args:
        nu:  (num_envs, 3) body velocities [u, v, r]
        psi: (num_envs,)   heading angle

    Returns:
        eta_dot: (num_envs, 3) world-frame derivatives [ẋ, ẏ, ψ̇]
    """
    u = nu[:, 0]
    v = nu[:, 1]
    r = nu[:, 2]
    cos_psi = torch.cos(psi)
    sin_psi = torch.sin(psi)

    x_dot = u * cos_psi - v * sin_psi
    y_dot = u * sin_psi + v * cos_psi
    psi_dot = r

    return torch.stack([x_dot, y_dot, psi_dot], dim=-1)


def wrap_angle(angle: Tensor) -> Tensor:
    """Wrap angle tensor to (−π, π]."""
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


# ---------------------------------------------------------------------------
# RK4 integrator
# ---------------------------------------------------------------------------

def integrate_rk4(
    state: Tensor,
    f: Callable[[Tensor], Tensor],
    dt: float,
) -> Tensor:
    """4th-order Runge-Kutta integration step.

    Args:
        state: (num_envs, state_dim) current state
        f:     derivative function  f(state) → state_dot
        dt:    timestep (s)

    Returns:
        next_state: (num_envs, state_dim)
    """
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_euler(
    state: Tensor,
    f: Callable[[Tensor], Tensor],
    dt: float,
) -> Tensor:
    """First-order Euler integration step (fast, less accurate)."""
    return state + dt * f(state)


# ---------------------------------------------------------------------------
# 3-DOF Rigid body integrator
# ---------------------------------------------------------------------------

class RigidBody3DOF:
    """Manages the 3-DOF vessel state and integrates dynamics.

    State layout (last dim):
        0: x      (m)
        1: y      (m)
        2: psi    (rad)
        3: u      (m/s)
        4: v      (m/s)
        5: r      (rad/s)
    """

    STATE_DIM: int = 6

    def __init__(
        self,
        cfg: BlueboatDynamicsConfig,
        num_envs: int,
        device: str = "cuda",
        use_rk4: bool = True,
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.use_rk4 = use_rk4

        self.hydro = HydrodynamicsModel(cfg, num_envs, device)

        # Current state tensor – mutable, lives on GPU
        self.state: Tensor = torch.zeros(
            num_envs, self.STATE_DIM, device=device
        )

        # Cache last control wrench for logging / smoothness penalty
        self.tau_last: Tensor = torch.zeros(num_envs, 3, device=device)

    # ------------------------------------------------------------------
    # State accessors (views, no copy)
    # ------------------------------------------------------------------

    @property
    def pos(self) -> Tensor:
        """(num_envs, 2) world position [x, y]."""
        return self.state[:, :2]

    @property
    def psi(self) -> Tensor:
        """(num_envs,) heading."""
        return self.state[:, 2]

    @property
    def nu(self) -> Tensor:
        """(num_envs, 3) body velocities [u, v, r]."""
        return self.state[:, 3:]

    @property
    def surge(self) -> Tensor:
        return self.state[:, 3]

    @property
    def sway(self) -> Tensor:
        return self.state[:, 4]

    @property
    def yaw_rate(self) -> Tensor:
        return self.state[:, 5]

    # ------------------------------------------------------------------
    # State derivative function  (pure function for RK4)
    # ------------------------------------------------------------------

    def _state_dot(
        self,
        state: Tensor,
        tau: Tensor,
        tau_env: Tensor,
        nu_current: Tensor | None = None,
    ) -> Tensor:
        """Compute full 6-DOF state derivative.

        Args:
            state:      (num_envs, 6)
            tau:        (num_envs, 3) control wrench
            tau_env:    (num_envs, 3) non-current env disturbance wrench
            nu_current: (num_envs, 3) optional current velocity in body frame

        Returns:
            state_dot: (num_envs, 6)
        """
        psi = state[:, 2]
        nu_cur = state[:, 3:]

        eta_dot = body_to_world(nu_cur, psi)
        nu_dot = self.hydro.compute_nu_dot(nu_cur, tau, tau_env, nu_current)

        return torch.cat([eta_dot, nu_dot], dim=-1)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        tau: Tensor,
        tau_env: Tensor,
        dt: float,
        nu_current: Tensor | None = None,
    ) -> None:
        """Advance state by one timestep.

        Args:
            tau:        (num_envs, 3) control wrench [X, Y, N]
            tau_env:    (num_envs, 3) non-current disturbance wrench
            dt:         timestep (s)
            nu_current: (num_envs, 3) optional current velocity in body frame;
                        when provided, damping is computed using water-relative
                        velocity (correct Fossen formulation)
        """
        self.tau_last = tau.clone()

        if self.use_rk4:
            def f(s: Tensor) -> Tensor:
                return self._state_dot(s, tau, tau_env, nu_current)

            self.state = integrate_rk4(self.state, f, dt)
        else:
            self.state = integrate_euler(
                self.state,
                lambda s: self._state_dot(s, tau, tau_env, nu_current),
                dt,
            )

        # Keep heading in (−π, π]
        self.state[:, 2] = wrap_angle(self.state[:, 2])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: Tensor, init_state: Tensor) -> None:
        """Reset specified environments to given initial states.

        Args:
            env_ids:    1D tensor of environment indices to reset
            init_state: (len(env_ids), 6) initial states
        """
        self.state[env_ids] = init_state
        self.tau_last[env_ids] = 0.0

    def reset_all(self) -> None:
        """Reset all environments to zero state."""
        self.state.zero_()
        self.tau_last.zero_()

    # ------------------------------------------------------------------
    # Velocity limit clamp (safety)
    # ------------------------------------------------------------------

    def clamp_velocities(
        self,
        max_surge: float,
        max_sway: float,
        max_yaw_rate: float,
    ) -> None:
        """Clip body velocities to prevent numerical blow-up."""
        self.state[:, 3].clamp_(-max_surge, max_surge)
        self.state[:, 4].clamp_(-max_sway, max_sway)
        self.state[:, 5].clamp_(-max_yaw_rate, max_yaw_rate)
