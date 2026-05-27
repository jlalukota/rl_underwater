"""
BlueboatEnv — primary HydraLab-USV reinforcement learning environment.

Phase 2 additions
─────────────────
- Correct relative-velocity current formulation (damping uses nu_rel = nu - nu_c)
- BoundaryReward module replaces inline penalty
- EpisodeStats accumulator for per-episode metrics
- Isaac Sim scene wired through isaac_scene helpers
- Configurable trajectory type (sinusoidal / lemniscate / circle / spline)

Architecture overview
─────────────────────
Dual execution mode:

1. **Isaac Lab mode** (production)
   Inherits from `omni.isaac.lab.envs.DirectRLEnv`.  Marine physics are
   computed by our custom dynamics layer; Isaac Sim provides rendering.

2. **Standalone mode** (development / CI)
   Falls back to gymnasium.Env-compatible base when Isaac Sim is absent.

State layout  (6-DOF NED planar)
──────────────────────────────────
    idx  symbol  unit  description
    0    x       m     world X (North)
    1    y       m     world Y (East)
    2    ψ       rad   heading (0 = North, CW positive)
    3    u       m/s   surge  (body +X)
    4    v       m/s   sway   (body +Y, positive starboard)
    5    r       rad/s yaw rate

Observation vector  (obs_dim = 10)
───────────────────────────────────
    [x_norm, y_norm, sin(ψ), cos(ψ), u_norm, v_norm, r_norm,
     Δx_goal_norm, Δy_goal_norm, dist_norm]

Heading is encoded as (sin ψ, cos ψ) to avoid the ±π discontinuity.

Action space  (2-DOF differential thrust)
──────────────────────────────────────────
    Box(low=-1, high=1, shape=(2,))  [a_left, a_right]
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from configs.blueboat_cfg import HydraLabConfig
from dynamics.rigid_body import RigidBody3DOF
from dynamics.thrust_model import DifferentialThrustModel
from disturbances.wind import WindDisturbance
from disturbances.current import CurrentDisturbance
from disturbances.waves import WaveDisturbance
from rewards.tracking import TrackingReward
from rewards.heading import HeadingReward, _wrap_angle
from rewards.smoothness import SmoothnessReward
from rewards.energy import EnergyReward
from rewards.boundary import BoundaryReward
from randomization.domain_randomization import DomainRandomizer
from tasks.trajectory_tracking import TrajectoryTrackingTask, TrajectoryType
from envs.episode_stats import EpisodeStats

# ---------------------------------------------------------------------------
# Conditional Isaac Lab import
# ---------------------------------------------------------------------------

try:
    from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
    import omni.isaac.lab.sim as sim_utils
    from envs.isaac_scene import (
        setup_water_surface,
        setup_lighting,
        build_vessel_cfg,
        build_goal_marker_cfg,
        env_origin,
        update_vessel_poses,
    )
    _ISAAC_LAB_AVAILABLE = True
except ImportError:
    _ISAAC_LAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Gymnasium fallback base class
# ---------------------------------------------------------------------------

if not _ISAAC_LAB_AVAILABLE:
    import gymnasium as gym
    from gymnasium import spaces

    class _GymBase:
        """Minimal gymnasium-compatible base when Isaac Lab is unavailable."""

        def __init__(self, cfg: HydraLabConfig) -> None:
            ec = cfg.env
            oc = cfg.obs
            self.num_envs: int = ec.num_envs
            self.device: str = cfg.device
            self.dt: float = ec.dt
            self.decimation: int = ec.decimation

            self.observation_space = spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(oc.obs_dim,),
                dtype="float32",
            )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype="float32",
            )

        def step(self, action): ...
        def reset(self, seed=None, options=None): ...
        def close(self): ...

    _EnvBase = _GymBase
else:
    _EnvBase = object


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class BlueboatEnv(_EnvBase):
    """
    Vectorised 3-DOF BlueBoat RL environment.

    Supports:
        - N parallel environments via GPU tensor operations
        - Isaac Lab DirectRLEnv API (when Isaac Sim is available)
        - Standalone gymnasium API (for development / testing)
        - PPO via Stable-Baselines3 through gymnasium wrappers
    """

    OBS_DIM = 10
    ACT_DIM = 2

    def __init__(
        self,
        cfg: Optional[HydraLabConfig] = None,
        render_mode: Optional[str] = None,
        traj_type: TrajectoryType = TrajectoryType.SINUSOIDAL,
        **kwargs: Any,
    ) -> None:
        self.cfg = cfg or HydraLabConfig()
        self._traj_type = traj_type

        if _ISAAC_LAB_AVAILABLE:
            super().__init__(self.cfg, render_mode, **kwargs)
        else:
            super().__init__(self.cfg)

        self._init_modules()

    # ------------------------------------------------------------------
    # Module initialisation
    # ------------------------------------------------------------------

    def _init_modules(self) -> None:
        cfg = self.cfg
        ec = cfg.env
        rc = cfg.reward
        N = self.num_envs
        dev = self.device

        # Dynamics
        self.body = RigidBody3DOF(cfg.dynamics, N, dev, use_rk4=True)
        self.thrust_model = DifferentialThrustModel(cfg.thrust, cfg.dynamics, N, dev)

        # Disturbances
        self.wind = WindDisturbance(cfg.disturbance, N, dev)
        self.current = CurrentDisturbance(cfg.disturbance, N, dev)
        self.waves = WaveDisturbance(cfg.disturbance, N, dev)

        # Task
        self.task = TrajectoryTrackingTask(ec, N, dev, traj_type=self._traj_type)

        # Reward modules
        self.r_tracking = TrackingReward(rc.tracking_weight, rc.tracking_sigma, dev)
        self.r_heading = HeadingReward(rc.heading_weight, rc.heading_sigma, dev)
        self.r_smoothness = SmoothnessReward(
            rc.smoothness_weight, rc.smoothness_delta_clip, dev
        )
        self.r_energy = EnergyReward(rc.energy_weight, dev)
        self.r_boundary = BoundaryReward(
            weight=rc.boundary_weight,
            workspace_radius=ec.workspace_radius,
            margin=rc.boundary_margin,
            hard_penalty=abs(rc.crash_penalty),
            device=dev,
        )

        # Domain randomizer
        self.randomizer = DomainRandomizer(cfg.randomization, ec, N, dev)

        # Episode statistics tracker
        self.stats = EpisodeStats(N, dev)

        # Per-step bookkeeping
        self.episode_length = torch.zeros(N, dtype=torch.long, device=dev)
        self.prev_action = torch.zeros(N, 2, device=dev)
        self._current_forces = torch.zeros(N, 2, device=dev)  # thruster lag state

        # Cached action (set by _pre_physics_step for multi-substep compatibility)
        self._raw_actions = torch.zeros(N, 2, device=dev)

    # ------------------------------------------------------------------
    # Isaac Lab scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        """Configure Isaac Sim scene — flat water surface + vessel assets."""
        if not _ISAAC_LAB_AVAILABLE:
            return

        setup_water_surface("/World/WaterSurface")
        setup_lighting("/World/SkyLight")

        # Register one vessel rigid body per environment
        # (Isaac Lab's clone_environments duplicates the base prim)
        vessel_cfg = build_vessel_cfg(env_idx=0)
        if vessel_cfg is not None:
            self._vessels = vessel_cfg.class_type(vessel_cfg)
            self.scene.articulations["vessel"] = self._vessels

        # Goal sphere markers
        # TODO: register as visual prims updated each step from task.target_pos

        self.scene.clone_environments(copy_from_source=False)

    # ------------------------------------------------------------------
    # Isaac Lab physics interface
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: Tensor) -> None:
        """Cache clamped actions before physics substeps."""
        self._raw_actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """Integrate marine dynamics for one physics substep."""
        dt = self.dt
        state = self.body.state

        # Thruster forces (with first-order actuator lag)
        self._current_forces = self.thrust_model.apply_lag(
            self._current_forces,
            self.thrust_model.normalized_to_force(self._raw_actions),
            dt,
        )
        tau = self.thrust_model.forces_to_wrench(self._current_forces)

        # Non-current disturbances (wind + waves) → body wrench
        tau_env = self.wind.compute(state) + self.waves.compute(state)
        # Note: current.compute() returns zeros — handled via relative velocity below

        # Current in body frame for relative-velocity damping (Phase 2)
        nu_current = self.current.get_body_frame_velocity(state)

        # Integrate dynamics with water-relative damping
        self.body.step(tau, tau_env, dt, nu_current=nu_current)
        self.body.clamp_velocities(
            self.cfg.env.max_surge,
            self.cfg.env.max_sway,
            self.cfg.env.max_yaw_rate,
        )

        # Advance disturbance internal state
        self.wind.step(dt)
        self.current.step(dt)
        self.waves.step(dt)

    # ------------------------------------------------------------------
    # Gymnasium step / reset  (standalone mode entry points)
    # ------------------------------------------------------------------

    def step(
        self, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        """Gymnasium-style step.

        Args:
            actions: (num_envs, 2) or (2,) for single-env

        Returns:
            obs, rewards, terminated, truncated, info

        For done environments the returned obs is the POST-RESET initial
        observation.  The pre-reset terminal observation is stored in
        info["_terminal_obs"] (shape: (n_done, obs_dim)) with corresponding
        environment indices in info["_terminal_obs_ids"], so that
        HydraLabVecEnvWrapper can populate the SB3 "terminal_observation"
        key needed for correct value bootstrap on episode boundaries.
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        actions = actions.to(self.device).clamp(-1.0, 1.0)

        self._pre_physics_step(actions)
        for _ in range(self.decimation):
            self._apply_action()

        self.task.step(self.cfg.env.policy_dt)

        rewards = self._compute_rewards(actions)
        terminated, truncated = self._get_dones()
        done_mask = terminated | truncated

        # Capture terminal observations BEFORE resetting done environments.
        # SB3 PPO needs these for value bootstrap at episode boundaries.
        terminal_obs = self._get_observations()

        # Update episode statistics
        dist = self.r_tracking.distance(self.body.pos, self.task.target_pos)
        boundary_flag = self.r_boundary.is_violated(self.body.pos)
        self.stats.update(rewards, dist, boundary_flag)
        episode_summaries = self.stats.collect_done(done_mask)

        # Bookkeeping
        self.episode_length += 1
        self.prev_action = actions.clone()

        # Reset done environments and build post-reset observation buffer.
        # Non-done envs retain their terminal obs as the next obs.
        done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._reset_idx(done_ids)
            post_reset_obs = self._get_observations()
            obs = terminal_obs.clone()
            obs[done_ids] = post_reset_obs[done_ids]
        else:
            obs = terminal_obs

        info = self._collect_info(episode_summaries)
        if done_ids.numel() > 0:
            info["_terminal_obs_ids"] = done_ids                   # (n_done,)
            info["_terminal_obs"] = terminal_obs[done_ids]         # (n_done, obs_dim)

        return obs, rewards, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[Tensor, dict]:
        if seed is not None:
            torch.manual_seed(seed)
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(all_ids)
        obs = self._get_observations()
        return obs, {}

    # ------------------------------------------------------------------
    # Reset implementation
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Tensor) -> None:
        if env_ids.numel() == 0:
            return

        init_states = self.randomizer.sample_initial_states(env_ids)
        self.body.reset(env_ids, init_states)

        self.randomizer.randomize_hydrodynamics(env_ids, self.body.hydro)
        self.randomizer.randomize_thrusters(env_ids, self.thrust_model)
        self.randomizer.seed_disturbances(
            env_ids, self.wind, self.current, self.waves
        )

        self.task.reset(env_ids)

        self.episode_length[env_ids] = 0
        self.prev_action[env_ids] = 0.0
        self._current_forces[env_ids] = 0.0
        self.stats.reset(env_ids)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> Tensor:
        """Build normalised observation tensor.

        Layout:
            0,1  : x_norm, y_norm
            2,3  : sin(ψ), cos(ψ)          ← avoids ±π discontinuity
            4    : u_norm
            5    : v_norm
            6    : r_norm
            7,8  : Δx_to_goal_norm, Δy_to_goal_norm
            9    : dist_to_goal_norm
        """
        oc = self.cfg.obs
        state = self.body.state
        pos = state[:, :2]
        psi = state[:, 2]
        u, v, r = state[:, 3], state[:, 4], state[:, 5]

        goal = self.task.target_pos
        delta_goal = goal - pos
        dist = delta_goal.norm(dim=-1, keepdim=True)

        obs = torch.cat(
            [
                pos / oc.pos_scale,
                torch.sin(psi).unsqueeze(-1),
                torch.cos(psi).unsqueeze(-1),
                (u / oc.vel_scale).unsqueeze(-1),
                (v / oc.vel_scale).unsqueeze(-1),
                (r / oc.heading_scale).unsqueeze(-1),
                delta_goal / oc.pos_scale,
                dist / oc.dist_scale,
            ],
            dim=-1,
        )

        if oc.add_noise:
            obs = self.randomizer.add_observation_noise(
                obs,
                pos_noise_std=oc.position_noise_std / oc.pos_scale,
                vel_noise_std=oc.velocity_noise_std / oc.vel_scale,
                heading_noise_std=oc.heading_noise_std,
                yaw_rate_noise_std=oc.yaw_rate_noise_std,
            )

        return obs

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _compute_rewards(self, actions: Tensor) -> Tensor:
        """Composite reward signal.

        R = w_track * r_track
          + w_head  * r_head
          - w_smooth * p_smooth
          - w_energy * p_energy
          - w_bound  * p_bound
        """
        state = self.body.state
        pos = state[:, :2]
        psi = state[:, 2]

        goal_pos = self.task.target_pos
        goal_heading = self.task.target_heading

        r_track = self.r_tracking.weighted(pos=pos, goal_pos=goal_pos)
        r_head = self.r_heading.weighted(psi=psi, goal_heading=goal_heading)
        p_smooth = self.r_smoothness.weighted(
            action=actions, prev_action=self.prev_action
        )
        p_energy = self.r_energy.weighted(action=actions)
        p_bound = self.r_boundary.weighted(pos=pos)

        return r_track + r_head - p_smooth - p_energy - p_bound

    # ------------------------------------------------------------------
    # Done conditions
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[Tensor, Tensor]:
        ec = self.cfg.env
        terminated = (
            self.r_boundary.is_violated(self.body.pos)
            | self.body.state.isnan().any(dim=-1)
            | self.body.state.isinf().any(dim=-1)
        )
        truncated = self.episode_length >= ec.episode_length_steps
        return terminated, truncated

    # ------------------------------------------------------------------
    # Isaac Lab reward interface
    # ------------------------------------------------------------------

    def _get_rewards(self) -> Tensor:
        return self._compute_rewards(self._raw_actions)

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _collect_info(
        self,
        episode_summaries: list[dict],
    ) -> dict[str, Any]:
        pos = self.body.pos
        dist = self.r_tracking.distance(pos, self.task.target_pos)
        info: dict[str, Any] = {
            "dist_to_goal": dist.mean().item(),
            "surge_mean": self.body.surge.mean().item(),
            "wind_speed_mean": self.wind.wind_speed.mean().item(),
            "current_speed_mean": self.current.speed.mean().item(),
            "episode_length_mean": self.episode_length.float().mean().item(),
        }
        if episode_summaries:
            agg = self.stats.summary(last_n=len(episode_summaries))
            info.update({f"ep/{k}": v for k, v in agg.items()})
        return info

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def close(self) -> None:
        pass
