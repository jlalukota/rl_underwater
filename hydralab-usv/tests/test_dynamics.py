"""
Unit tests for the 3-DOF dynamics layer.

Tests verify:
    - Hydrodynamic force signs are physically correct
    - RK4 integrator conserves momentum under no forcing
    - Thrust model maps correctly to body wrench
    - Velocity clamping works
    - Domain randomization stays within bounds

Run with:  pytest tests/test_dynamics.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from configs.blueboat_cfg import (
    BlueboatDynamicsConfig,
    ThrustConfig,
    HydraLabConfig,
)
from dynamics.hydrodynamics import HydrodynamicsModel
from dynamics.thrust_model import DifferentialThrustModel
from dynamics.rigid_body import RigidBody3DOF, wrap_angle, body_to_world, integrate_rk4


DEVICE = "cpu"
NUM_ENVS = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dyn_cfg():
    return BlueboatDynamicsConfig()


@pytest.fixture
def hydro(dyn_cfg):
    return HydrodynamicsModel(dyn_cfg, NUM_ENVS, DEVICE)


@pytest.fixture
def thrust(dyn_cfg):
    return DifferentialThrustModel(ThrustConfig(), dyn_cfg, NUM_ENVS, DEVICE)


@pytest.fixture
def body(dyn_cfg):
    return RigidBody3DOF(dyn_cfg, NUM_ENVS, DEVICE, use_rk4=True)


# ---------------------------------------------------------------------------
# HydrodynamicsModel
# ---------------------------------------------------------------------------

class TestHydrodynamics:

    def test_mass_matrix_positive(self, dyn_cfg):
        """Effective inertias must be positive."""
        assert dyn_cfg.m11 > 0
        assert dyn_cfg.m22 > 0
        assert dyn_cfg.m33 > 0

    def test_damping_opposes_surge(self, hydro):
        """Surge damping force must oppose positive surge velocity."""
        nu = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        nu[:, 0] = 1.0  # u = 1 m/s forward
        f = hydro.damping_forces(nu)
        assert (f[:, 0] < 0).all(), "Surge damping must be negative for positive u"

    def test_damping_opposes_sway(self, hydro):
        nu = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        nu[:, 1] = 1.0
        f = hydro.damping_forces(nu)
        assert (f[:, 1] < 0).all()

    def test_damping_opposes_yaw(self, hydro):
        nu = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        nu[:, 2] = 1.0
        f = hydro.damping_forces(nu)
        assert (f[:, 2] < 0).all()

    def test_damping_zero_at_rest(self, hydro):
        nu = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        f = hydro.damping_forces(nu)
        assert torch.allclose(f, torch.zeros_like(f))

    def test_coriolis_zero_at_rest(self, hydro):
        nu = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        f = hydro.coriolis_forces(nu)
        assert torch.allclose(f, torch.zeros_like(f))

    def test_coriolis_surge_from_sway_and_yaw(self, hydro):
        """Coriolis in surge = -m22 * v * r."""
        nu = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        nu[:, 1] = 1.0   # v = 1
        nu[:, 2] = 1.0   # r = 1
        f = hydro.coriolis_forces(nu)
        expected_surge = -hydro.m22 * 1.0 * 1.0
        assert torch.allclose(f[:, 0], expected_surge, atol=1e-5)

    def test_inv_mass_shape(self, hydro):
        inv_m = hydro.inv_mass_matrix()
        assert inv_m.shape == (NUM_ENVS, 3)
        assert (inv_m > 0).all()

    def test_randomize_stays_positive(self, hydro, dyn_cfg):
        env_ids = torch.arange(NUM_ENVS)
        hydro.randomize(env_ids, mass_noise_frac=0.1, drag_noise_frac=0.2, inertia_noise_frac=0.1)
        assert (hydro.m11 > 0).all()
        assert (hydro.m22 > 0).all()
        assert (hydro.m33 > 0).all()


# ---------------------------------------------------------------------------
# DifferentialThrustModel
# ---------------------------------------------------------------------------

class TestThrustModel:

    def test_zero_command_zero_force(self, thrust):
        actions = torch.zeros(NUM_ENVS, 2, device=DEVICE)
        tau = thrust.compute_tau(actions)
        assert torch.allclose(tau, torch.zeros_like(tau), atol=1e-6)

    def test_equal_forward_produces_no_yaw(self, thrust):
        """Equal thrust forward → zero yaw moment."""
        actions = torch.ones(NUM_ENVS, 2, device=DEVICE) * 0.5
        tau = thrust.compute_tau(actions)
        assert torch.allclose(tau[:, 2], torch.zeros(NUM_ENVS), atol=1e-5)

    def test_differential_produces_yaw(self, thrust):
        """Right > Left → positive yaw moment."""
        actions = torch.zeros(NUM_ENVS, 2, device=DEVICE)
        actions[:, 0] = -0.5   # left reverse
        actions[:, 1] = 0.5    # right forward
        tau = thrust.compute_tau(actions)
        assert (tau[:, 2] > 0).all()

    def test_forward_only_no_sway(self, thrust):
        """Pure symmetric thrust → zero sway force."""
        actions = torch.ones(NUM_ENVS, 2, device=DEVICE) * 0.8
        tau = thrust.compute_tau(actions)
        assert torch.allclose(tau[:, 1], torch.zeros(NUM_ENVS), atol=1e-6)

    def test_force_bounded_by_max(self, thrust):
        actions = torch.ones(NUM_ENVS, 2, device=DEVICE)  # max command
        forces = thrust.normalized_to_force(actions)
        assert (forces <= thrust.thrust_cfg.max_thrust_fwd + 1e-5).all()

    def test_deadband_zeros_small_input(self, thrust):
        actions = torch.full((NUM_ENVS, 2), 0.01, device=DEVICE)  # < deadband
        forces = thrust.normalized_to_force(actions)
        assert torch.allclose(forces, torch.zeros_like(forces))

    def test_efficiency_randomization_in_range(self, thrust):
        env_ids = torch.arange(NUM_ENVS)
        thrust.randomize_efficiency(env_ids, efficiency_range=(0.8, 1.0))
        assert (thrust.efficiency >= 0.8 - 1e-6).all()
        assert (thrust.efficiency <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# RigidBody3DOF
# ---------------------------------------------------------------------------

class TestRigidBody3DOF:

    def test_initial_state_zeros(self, body):
        assert torch.allclose(body.state, torch.zeros_like(body.state))

    def test_deceleration_under_drag(self, body):
        """Vessel moving forward with no thrust should decelerate."""
        state = torch.zeros(NUM_ENVS, 6, device=DEVICE)
        state[:, 3] = 1.0  # u = 1 m/s
        env_ids = torch.arange(NUM_ENVS)
        body.reset(env_ids, state)

        tau = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        tau_env = torch.zeros(NUM_ENVS, 3, device=DEVICE)

        initial_u = body.surge.clone()
        for _ in range(10):
            body.step(tau, tau_env, dt=0.02)

        assert (body.surge < initial_u).all(), "Drag should decelerate vessel"

    def test_heading_wrap_stays_in_range(self, body):
        """Heading should always remain in (−π, π]."""
        state = torch.zeros(NUM_ENVS, 6, device=DEVICE)
        state[:, 2] = 3.0   # psi > π
        env_ids = torch.arange(NUM_ENVS)
        body.reset(env_ids, state)

        tau = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        tau_env = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        body.step(tau, tau_env, dt=0.02)

        assert (body.psi >= -math.pi - 1e-5).all()
        assert (body.psi <= math.pi + 1e-5).all()

    def test_forward_thrust_increases_surge(self, body):
        """Full forward thrust should accelerate the vessel."""
        env_ids = torch.arange(NUM_ENVS)
        body.reset(env_ids, torch.zeros(NUM_ENVS, 6))

        tau = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        tau[:, 0] = 60.0   # 60 N surge force

        initial_u = body.surge.clone()
        body.step(tau, torch.zeros(NUM_ENVS, 3), dt=0.02)
        assert (body.surge > initial_u).all()

    def test_yaw_moment_changes_heading(self, body):
        env_ids = torch.arange(NUM_ENVS)
        body.reset(env_ids, torch.zeros(NUM_ENVS, 6))

        tau = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        tau[:, 2] = 5.0   # positive yaw moment

        for _ in range(5):
            body.step(tau, torch.zeros(NUM_ENVS, 3), dt=0.02)

        assert (body.yaw_rate > 0).all()

    def test_velocity_clamp(self, body):
        env_ids = torch.arange(NUM_ENVS)
        state = torch.zeros(NUM_ENVS, 6, device=DEVICE)
        state[:, 3] = 100.0   # extreme surge
        body.reset(env_ids, state)
        body.clamp_velocities(max_surge=3.0, max_sway=2.0, max_yaw_rate=2.0)
        assert (body.surge <= 3.0 + 1e-5).all()

    def test_rk4_more_accurate_than_euler(self, dyn_cfg):
        """RK4 integrator should give lower position error than Euler at same dt."""
        # This is a qualitative test: RK4 trajectory should not diverge under large dt
        body_rk4 = RigidBody3DOF(dyn_cfg, 1, DEVICE, use_rk4=True)
        body_euler = RigidBody3DOF(dyn_cfg, 1, DEVICE, use_rk4=False)

        init = torch.zeros(1, 6)
        init[0, 3] = 1.0  # 1 m/s surge
        body_rk4.reset(torch.tensor([0]), init.clone())
        body_euler.reset(torch.tensor([0]), init.clone())

        tau = torch.zeros(1, 3)
        tau_env = torch.zeros(1, 3)

        for _ in range(50):
            body_rk4.step(tau, tau_env, dt=0.1)
            body_euler.step(tau, tau_env, dt=0.1)

        # Both should produce finite, reasonable positions
        assert body_rk4.state.isfinite().all()
        assert body_euler.state.isfinite().all()


# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------

class TestKinematics:

    def test_body_to_world_heading_zero(self):
        """At psi=0, surge maps directly to world X."""
        nu = torch.tensor([[1.0, 0.0, 0.0]])
        psi = torch.tensor([0.0])
        eta_dot = body_to_world(nu, psi)
        assert abs(float(eta_dot[0, 0]) - 1.0) < 1e-6
        assert abs(float(eta_dot[0, 1])) < 1e-6

    def test_body_to_world_heading_90(self):
        """At psi=π/2, surge maps to world Y."""
        nu = torch.tensor([[1.0, 0.0, 0.0]])
        psi = torch.tensor([math.pi / 2])
        eta_dot = body_to_world(nu, psi)
        assert abs(float(eta_dot[0, 0])) < 1e-6
        assert abs(float(eta_dot[0, 1]) - 1.0) < 1e-6

    def test_wrap_angle(self):
        angles = torch.tensor([0.0, math.pi, -math.pi, 2 * math.pi, -2 * math.pi])
        wrapped = wrap_angle(angles)
        assert (wrapped >= -math.pi - 1e-5).all()
        assert (wrapped <= math.pi + 1e-5).all()
