"""
Configuration dataclasses for HydraLab-USV.

All physical parameters are based on the BlueRobotics BlueBoat catamaran.
References:
  - Fossen, T. I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control.
  - BlueRobotics BlueBoat specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Vehicle physical model
# ---------------------------------------------------------------------------

@dataclass
class BlueboatDynamicsConfig:
    """Rigid-body and hydrodynamic parameters for BlueBoat-style catamaran.

    Sign convention follows Fossen (2011):
      - Xu, Yv, Nr are negative (drag opposes motion)
      - Xuu, Yvv, Nrr are negative (quadratic drag)
      - Added-mass coefficients (Xudot, etc.) are negative
    """

    # Hull inertia
    mass: float = 15.0           # kg  (hull + payload + batteries)
    inertia_z: float = 3.5       # kg·m²  (yaw moment of inertia)
    x_g: float = 0.0             # m    (CoG offset from body origin, assume symmetric)

    # Thruster geometry
    beam: float = 0.57           # m    (lateral separation between port/stbd thrusters)

    # Added-mass coefficients (all negative per Fossen convention)
    Xudot: float = -1.5          # kg
    Yvdot: float = -10.0         # kg
    Yrdot: float = 0.0           # kg·m  (coupled sway–yaw added mass)
    Nvdot: float = 0.0           # kg·m
    Nrdot: float = -1.8          # kg·m²

    # Linear damping coefficients (all negative)
    Xu: float = -3.0             # N·s/m
    Yv: float = -8.0             # N·s/m
    Yr: float = 0.0              # N·s/rad (coupled sway–yaw damping)
    Nv: float = 0.0              # N·m·s/m
    Nr: float = -2.5             # N·m·s/rad

    # Quadratic damping coefficients (all negative)
    Xuu: float = -5.0            # N·s²/m²
    Yvv: float = -20.0           # N·s²/m²
    Nrr: float = -5.0            # N·m·s²/rad²

    @property
    def m11(self) -> float:
        """Effective surge inertia (rigid body + added mass)."""
        return self.mass - self.Xudot

    @property
    def m22(self) -> float:
        """Effective sway inertia."""
        return self.mass - self.Yvdot

    @property
    def m33(self) -> float:
        """Effective yaw inertia."""
        return self.inertia_z - self.Nrdot


# ---------------------------------------------------------------------------
# Thruster model
# ---------------------------------------------------------------------------

@dataclass
class ThrustConfig:
    """Actuator parameters for a single T200-class thruster."""

    max_thrust_fwd: float = 30.0    # N  (forward, ~3.06 kgf at 12V)
    max_thrust_rev: float = 22.0    # N  (reverse is asymmetric on T200)
    time_constant: float = 0.1      # s  (first-order lag on thrust response)
    deadband: float = 0.02          # normalized input dead-zone

    # TODO: replace with polynomial fit from BlueRobotics T200 thrust curves
    # for higher-fidelity sim-to-real transfer


# ---------------------------------------------------------------------------
# Simulation environment
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentConfig:
    """Isaac Lab vectorized environment parameters."""

    num_envs: int = 1024
    dt: float = 0.02              # physics timestep (s) → 50 Hz
    decimation: int = 4           # physics steps per policy decision
    episode_length_s: float = 60.0  # seconds per episode

    # Workspace / safety bounds
    workspace_radius: float = 50.0  # m  – environments reset beyond this
    max_surge: float = 3.0          # m/s
    max_sway: float = 2.0           # m/s
    max_yaw_rate: float = 2.0       # rad/s

    # Goal tolerances
    goal_radius: float = 0.5        # m  – waypoint considered reached
    heading_tol: float = 0.1        # rad

    @property
    def policy_dt(self) -> float:
        return self.dt * self.decimation

    @property
    def episode_length_steps(self) -> int:
        return int(self.episode_length_s / self.policy_dt)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@dataclass
class ObservationConfig:
    """Observation space definition and sensor noise parameters."""

    obs_dim: int = 10  # [x, y, psi, u, v, r, goal_x, goal_y, heading_err, dist]

    add_noise: bool = True
    position_noise_std: float = 0.02    # m
    velocity_noise_std: float = 0.01    # m/s
    heading_noise_std: float = 0.005    # rad
    yaw_rate_noise_std: float = 0.002   # rad/s

    # Observation scaling (divide raw obs by these values before feeding policy)
    pos_scale: float = 50.0             # m
    vel_scale: float = 3.0              # m/s
    heading_scale: float = 3.14159      # rad
    dist_scale: float = 50.0            # m


# ---------------------------------------------------------------------------
# Disturbances
# ---------------------------------------------------------------------------

@dataclass
class DisturbanceConfig:
    """Environmental disturbance model parameters."""

    # Wind model
    wind_enabled: bool = True
    wind_speed_mean: float = 2.0         # m/s
    wind_speed_std: float = 0.5          # m/s  (Gaussian noise on top of mean)
    wind_update_period: float = 5.0      # s  (how often direction/speed changes)
    wind_drag_coeff: float = 0.8         # N/(m/s)² projected area × drag coeff

    # Surface current model
    current_enabled: bool = True
    current_speed_mean: float = 0.3      # m/s
    current_speed_std: float = 0.1       # m/s
    current_update_period: float = 10.0  # s

    # Stochastic wave injection
    wave_enabled: bool = True
    wave_force_std: float = 1.0          # N  (Gaussian surge/sway impulse std)
    wave_moment_std: float = 0.3         # N·m
    wave_bandwidth: float = 0.5          # Hz  (first-order Markov bandwidth)


# ---------------------------------------------------------------------------
# Domain randomization
# ---------------------------------------------------------------------------

@dataclass
class DomainRandomizationConfig:
    """Ranges for domain randomization applied at episode reset."""

    enabled: bool = True

    # Initial state scatter
    initial_pos_range: float = 5.0           # m  (uniform ±)
    initial_heading_range: float = 3.14159   # rad (uniform ±π)
    initial_vel_range: float = 0.0           # m/s (zero for V1)

    # Environmental randomization
    wind_speed_range: Tuple[float, float] = (0.0, 5.0)    # m/s
    wind_dir_range: Tuple[float, float] = (0.0, 6.2832)   # rad
    current_speed_range: Tuple[float, float] = (0.0, 1.0) # m/s
    current_dir_range: Tuple[float, float] = (0.0, 6.2832)

    # Parameter randomization (multiplicative noise)
    mass_noise_frac: float = 0.10            # ±10% of nominal mass
    drag_noise_frac: float = 0.20            # ±20% of nominal drag coefficients
    thrust_efficiency_range: Tuple[float, float] = (0.80, 1.00)
    inertia_noise_frac: float = 0.10


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Weights and shaping parameters for the composite reward function."""

    # Tracking
    tracking_weight: float = 2.0
    tracking_sigma: float = 1.0       # Gaussian shaping bandwidth (m)

    # Heading alignment
    heading_weight: float = 0.5
    heading_sigma: float = 0.5        # rad

    # Smoothness penalty
    smoothness_weight: float = 0.1
    smoothness_delta_clip: float = 2.0  # max delta-action considered

    # Energy penalty
    energy_weight: float = 0.05

    # Boundary penalty
    boundary_weight: float = 5.0
    boundary_margin: float = 5.0      # m  (soft boundary before hard reset)

    # Terminal bonuses / penalties
    success_bonus: float = 10.0
    crash_penalty: float = -20.0


# ---------------------------------------------------------------------------
# Top-level composite config
# ---------------------------------------------------------------------------

@dataclass
class HydraLabConfig:
    """Master configuration bundling all sub-configs."""

    dynamics: BlueboatDynamicsConfig = field(default_factory=BlueboatDynamicsConfig)
    thrust: ThrustConfig = field(default_factory=ThrustConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    reward: RewardConfig = field(default_factory=RewardConfig)

    seed: int = 42
    device: str = "cuda"
