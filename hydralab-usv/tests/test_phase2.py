"""
Phase 2 integration and unit tests.

Covers:
    - Relative-velocity current: damping uses nu_rel
    - BoundaryReward: correct penalty zones
    - EpisodeStats: accumulation and collection
    - Catmull-Rom spline trajectory: finite, smooth
    - YAML config round-trip
    - Benchmark runner (smoke test)
    - Evaluate with random policy (smoke test)
    - CurrentDisturbance: body-frame velocity interface

Run with:  pytest tests/test_phase2.py -v
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from configs.blueboat_cfg import HydraLabConfig
from dynamics.hydrodynamics import HydrodynamicsModel
from dynamics.rigid_body import RigidBody3DOF
from disturbances.current import CurrentDisturbance
from rewards.boundary import BoundaryReward
from envs.episode_stats import EpisodeStats
from envs.blueboat_env import BlueboatEnv
from tasks.trajectory_tracking import TrajectoryTrackingTask, TrajectoryType


DEVICE = "cpu"
N = 16


def _cfg(num_envs=N):
    c = HydraLabConfig()
    c.env.num_envs = num_envs
    c.device = DEVICE
    c.obs.add_noise = False
    return c


# ---------------------------------------------------------------------------
# Relative-velocity current  (Phase 2 hydrodynamics correctness)
# ---------------------------------------------------------------------------

class TestRelativeVelocityCurrent:

    def test_current_aligned_with_surge_reduces_drag(self):
        """A current flowing in the same direction as the vessel should
        reduce hydrodynamic drag, allowing the vessel to travel faster."""
        from configs.blueboat_cfg import BlueboatDynamicsConfig
        cfg = BlueboatDynamicsConfig()
        hydro_no_cur = HydrodynamicsModel(cfg, 1, DEVICE)
        hydro_with_cur = HydrodynamicsModel(cfg, 1, DEVICE)

        nu = torch.tensor([[1.0, 0.0, 0.0]])   # vessel moving at 1 m/s surge
        tau = torch.zeros(1, 3)
        tau_env = torch.zeros(1, 3)

        # No current
        nu_dot_no = hydro_no_cur.compute_nu_dot(nu, tau, tau_env, nu_current=None)

        # Current in the same direction as vessel (1 m/s surge)
        nu_current = torch.tensor([[0.8, 0.0, 0.0]])
        nu_dot_cur = hydro_with_cur.compute_nu_dot(nu, tau, tau_env, nu_current=nu_current)

        # With favourable current, drag is lower → deceleration is less negative
        assert float(nu_dot_cur[0, 0]) > float(nu_dot_no[0, 0]), \
            "Favourable current should reduce drag (less negative u_dot)"

    def test_opposing_current_increases_drag(self):
        """An opposing current increases effective drag."""
        from configs.blueboat_cfg import BlueboatDynamicsConfig
        cfg = BlueboatDynamicsConfig()
        hydro = HydrodynamicsModel(cfg, 1, DEVICE)

        nu = torch.tensor([[1.0, 0.0, 0.0]])
        tau = torch.zeros(1, 3)
        tau_env = torch.zeros(1, 3)
        nu_current = torch.tensor([[-0.5, 0.0, 0.0]])  # opposing current

        nu_dot_no = hydro.compute_nu_dot(nu, tau, tau_env)
        nu_dot_cur = hydro.compute_nu_dot(nu, tau, tau_env, nu_current=nu_current)

        assert float(nu_dot_cur[0, 0]) < float(nu_dot_no[0, 0]), \
            "Opposing current increases drag (more negative u_dot)"

    def test_current_disturbance_compute_returns_zeros(self):
        """CurrentDisturbance.compute() must return zeros in Phase 2."""
        from configs.blueboat_cfg import DisturbanceConfig
        cfg_d = DisturbanceConfig()
        cur = CurrentDisturbance(cfg_d, N, DEVICE)
        state = torch.zeros(N, 6)
        tau = cur.compute(state)
        assert torch.allclose(tau, torch.zeros(N, 3))

    def test_current_body_velocity_shape(self):
        from configs.blueboat_cfg import DisturbanceConfig
        cur = CurrentDisturbance(DisturbanceConfig(), N, DEVICE)
        state = torch.zeros(N, 6)
        nu_c = cur.get_body_frame_velocity(state)
        assert nu_c.shape == (N, 3)
        assert nu_c.isfinite().all()
        # Yaw component should always be zero
        assert torch.allclose(nu_c[:, 2], torch.zeros(N))

    def test_current_body_velocity_rotates_with_heading(self):
        """At psi=π/2, a north-flowing current maps to sway component."""
        from configs.blueboat_cfg import DisturbanceConfig
        cfg_d = DisturbanceConfig()
        cfg_d.current_speed_mean = 1.0
        cfg_d.current_speed_std = 0.0
        cur = CurrentDisturbance(cfg_d, 1, DEVICE)
        # Force a known current direction (north = +X world)
        cur._speed[:] = 1.0
        cur._direction[:] = 0.0   # θ=0 → pure X (north) current

        # Vessel heading = π/2 (facing east)
        state = torch.zeros(1, 6)
        state[0, 2] = math.pi / 2.0
        nu_c = cur.get_body_frame_velocity(state)

        # North current (+X world), vessel facing east (ψ=π/2, bow = +Y world).
        # Body-frame rotation: R(ψ)^T * V_world
        #   u_c = cos(π/2)*1 + sin(π/2)*0 = 0
        #   v_c = -sin(π/2)*1 + cos(π/2)*0 = -1
        # Negative v_c means the current flows from starboard-to-port
        # (which is correct: vessel faces east, current is northward, so
        #  the current hits the port side → negative v in body frame).
        assert abs(float(nu_c[0, 0])) < 0.01, "u_c should be ~0"
        assert abs(float(nu_c[0, 1]) + 1.0) < 0.01, "v_c should be ~−1 (port-side current)"


# ---------------------------------------------------------------------------
# BoundaryReward
# ---------------------------------------------------------------------------

class TestBoundaryReward:

    @pytest.fixture
    def br(self):
        return BoundaryReward(
            weight=1.0,
            workspace_radius=50.0,
            margin=5.0,
            hard_penalty=20.0,
            device=DEVICE,
        )

    def test_zero_inside_soft_zone(self, br):
        pos = torch.zeros(N, 2)   # all at origin, well inside
        p = br.compute(pos=pos)
        assert torch.allclose(p, torch.zeros(N))

    def test_penalty_grows_in_margin_zone(self, br):
        """At 46 m (inside margin zone of 50-5=45 m..50 m), penalty is positive."""
        pos = torch.zeros(N, 2)
        pos[:, 0] = 47.0   # 47 m > 45 m soft start
        p = br.compute(pos=pos)
        assert (p > 0).all()

    def test_hard_penalty_beyond_boundary(self, br):
        pos = torch.zeros(N, 2)
        pos[:, 0] = 55.0   # clearly outside
        p = br.compute(pos=pos)
        assert (p >= br.hard_penalty - 1e-4).all()

    def test_penalty_monotone_with_distance(self, br):
        dists = [0.0, 10.0, 44.0, 47.0, 52.0]
        prev = -1.0
        for d in dists:
            pos = torch.zeros(1, 2)
            pos[0, 0] = d
            p = float(br.compute(pos=pos)[0])
            assert p >= prev, f"Penalty not monotone at d={d}"
            prev = p

    def test_is_violated_correct(self, br):
        pos_inside = torch.zeros(N, 2)
        pos_inside[:, 0] = 49.0
        assert not br.is_violated(pos_inside).any()

        pos_outside = torch.zeros(N, 2)
        pos_outside[:, 0] = 51.0
        assert br.is_violated(pos_outside).all()


# ---------------------------------------------------------------------------
# EpisodeStats
# ---------------------------------------------------------------------------

class TestEpisodeStats:

    def test_accumulates_rewards(self):
        stats = EpisodeStats(N, DEVICE)
        rewards = torch.ones(N) * 2.0
        dist = torch.ones(N) * 0.5
        for _ in range(10):
            stats.update(rewards, dist)
        assert torch.allclose(stats.current_returns, torch.ones(N) * 20.0)

    def test_collect_done_returns_correct_env_ids(self):
        stats = EpisodeStats(N, DEVICE)
        rewards = torch.ones(N)
        dist = torch.zeros(N)
        for _ in range(5):
            stats.update(rewards, dist)

        done = torch.zeros(N, dtype=torch.bool)
        done[0] = True
        done[3] = True
        summaries = stats.collect_done(done)
        collected_ids = {s["env_id"] for s in summaries}
        assert collected_ids == {0, 3}

    def test_reset_clears_accumulators(self):
        stats = EpisodeStats(N, DEVICE)
        stats.update(torch.ones(N), torch.zeros(N))
        stats.reset(torch.tensor([0, 1, 2]))
        assert float(stats.current_returns[0]) == 0.0
        assert float(stats.current_returns[3]) == 1.0  # unreset

    def test_summary_correct_mean(self):
        stats = EpisodeStats(4, DEVICE)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        dist = torch.zeros(4)
        stats.update(rewards, dist)
        done = torch.ones(4, dtype=torch.bool)
        stats.collect_done(done)
        s = stats.summary()
        assert abs(s["mean_return"] - 2.5) < 1e-5

    def test_boundary_hits_counted(self):
        stats = EpisodeStats(N, DEVICE)
        rewards = torch.zeros(N)
        dist = torch.zeros(N)
        bflag = torch.zeros(N, dtype=torch.bool)
        bflag[0] = True
        stats.update(rewards, dist, bflag)
        assert int(stats._boundary_hits[0]) == 1
        assert int(stats._boundary_hits[1]) == 0


# ---------------------------------------------------------------------------
# Spline trajectory (Catmull-Rom)
# ---------------------------------------------------------------------------

class TestSplineTrajectory:

    @pytest.fixture
    def spline_task(self):
        cfg = _cfg().env
        return TrajectoryTrackingTask(
            cfg, N, DEVICE, traj_type=TrajectoryType.SPLINE
        )

    def test_target_pos_finite(self, spline_task):
        assert spline_task.target_pos.isfinite().all()
        assert spline_task.target_heading.isfinite().all()

    def test_target_advances(self, spline_task):
        pos_before = spline_task.target_pos.clone()
        for _ in range(10):
            spline_task.step(0.08)
        pos_after = spline_task.target_pos
        assert not torch.allclose(pos_before, pos_after)

    def test_spline_stays_in_reasonable_range(self, spline_task):
        for _ in range(100):
            spline_task.step(0.1)
        pos = spline_task.target_pos
        assert (pos.abs() < 100.0).all(), "Spline pos should stay bounded"

    def test_spline_wraps_without_nan(self, spline_task):
        """Step past a full loop and verify no NaN."""
        for _ in range(500):
            spline_task.step(0.1)
        assert spline_task.target_pos.isfinite().all()

    def test_reset_gives_different_trajectory(self, spline_task):
        pos_before = spline_task.target_pos.clone()
        spline_task.reset(torch.arange(N))
        assert not torch.allclose(pos_before, spline_task.target_pos)


# ---------------------------------------------------------------------------
# Phase 2 BlueboatEnv integration
# ---------------------------------------------------------------------------

class TestBlueboatEnvPhase2:

    @pytest.fixture
    def env(self):
        e = BlueboatEnv(_cfg(), traj_type=TrajectoryType.SINUSOIDAL)
        e.reset()
        return e

    def test_current_velocity_changes_dynamics(self):
        """Vessel with opposing current should decelerate faster."""
        cfg_high_cur = _cfg(num_envs=1)
        cfg_high_cur.disturbance.current_speed_mean = 2.0
        cfg_high_cur.disturbance.current_speed_std = 0.0
        cfg_high_cur.disturbance.wind_enabled = False
        cfg_high_cur.disturbance.wave_enabled = False
        cfg_high_cur.randomization.enabled = False

        cfg_no_cur = _cfg(num_envs=1)
        cfg_no_cur.disturbance.current_speed_mean = 0.0
        cfg_no_cur.disturbance.current_speed_std = 0.0
        cfg_no_cur.disturbance.wind_enabled = False
        cfg_no_cur.disturbance.wave_enabled = False
        cfg_no_cur.randomization.enabled = False

        env_cur = BlueboatEnv(cfg_high_cur)
        env_no = BlueboatEnv(cfg_no_cur)

        init = torch.zeros(1, 6)
        init[0, 3] = 1.5  # surge 1.5 m/s
        # Force current to oppose motion (direction = π, i.e. south/negative-x)
        import math
        env_cur.current._direction[:] = math.pi
        env_cur.current._speed[:] = 2.0

        env_cur.body.reset(torch.tensor([0]), init.clone())
        env_no.body.reset(torch.tensor([0]), init.clone())

        zero_action = torch.zeros(1, 2)
        for _ in range(20):
            env_cur.step(zero_action)
            env_no.step(zero_action)

        # Opposing current means more relative velocity → more drag → slower
        assert float(env_cur.body.surge[0]) < float(env_no.body.surge[0]), \
            "Opposing current should cause more deceleration"

    def test_episode_stats_populated_after_done(self, env):
        """Stats summary should contain data after episodes complete."""
        # Run enough steps to trigger a truncation
        for _ in range(env.cfg.env.episode_length_steps + 5):
            env.step(torch.zeros(N, 2))
        s = env.stats.summary()
        # At least some episodes should have completed
        assert s.get("n_episodes", 0) > 0 or len(env.stats._completed) > 0

    def test_boundary_reward_in_reward_computation(self, env):
        """Vessels far from origin should incur boundary penalty."""
        # Teleport vessels near boundary
        env.body.state[:, 0] = env.cfg.env.workspace_radius - 3.0
        rewards_near = env._compute_rewards(torch.zeros(N, 2))

        env.body.state[:, 0] = 0.0
        rewards_center = env._compute_rewards(torch.zeros(N, 2))

        assert (rewards_near < rewards_center).all(), \
            "Vessels near boundary should receive lower reward"

    def test_spline_env_runs(self):
        """BlueboatEnv with SPLINE trajectory completes steps without error."""
        env = BlueboatEnv(_cfg(), traj_type=TrajectoryType.SPLINE)
        obs, _ = env.reset()
        assert obs.isfinite().all()
        for _ in range(20):
            obs, rew, *_ = env.step(torch.zeros(N, 2))
        assert obs.isfinite().all()
        assert rew.isfinite().all()


# ---------------------------------------------------------------------------
# YAML config round-trip
# ---------------------------------------------------------------------------

def _yaml_available() -> bool:
    from configs.yaml_loader import _YAML_AVAILABLE
    return _YAML_AVAILABLE


class TestYamlConfig:

    def test_save_and_load_roundtrip(self):
        if not _yaml_available():
            pytest.skip("PyYAML not installed (pip install pyyaml)")
        from configs.yaml_loader import save_config, load_config

        cfg_orig = HydraLabConfig()
        cfg_orig.env.num_envs = 512
        cfg_orig.reward.tracking_weight = 3.5

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name

        save_config(cfg_orig, path)
        cfg_loaded = load_config(path)

        assert cfg_loaded.env.num_envs == 512
        assert abs(cfg_loaded.reward.tracking_weight - 3.5) < 1e-6
        Path(path).unlink()

    def test_override_applies(self):
        if not _yaml_available():
            pytest.skip("PyYAML not installed (pip install pyyaml)")
        from configs.yaml_loader import save_config, load_config

        cfg_orig = HydraLabConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        save_config(cfg_orig, path)

        cfg_loaded = load_config(path, overrides={"env.num_envs": 999})
        assert cfg_loaded.env.num_envs == 999
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Benchmark smoke test
# ---------------------------------------------------------------------------

class TestBenchmark:

    def test_benchmark_runs_and_returns_positive_sps(self):
        from training.benchmark import benchmark
        result = benchmark(num_envs=32, device="cpu", warmup_steps=5, measure_steps=20)
        assert result["sps"] > 0
        assert result["num_envs"] == 32
        assert result["ms_per_step"] > 0


# ---------------------------------------------------------------------------
# Evaluate smoke test (random policy, no trained model required)
# ---------------------------------------------------------------------------

class TestEvaluateRandomPolicy:

    def test_random_baseline_returns_dict(self):
        from training.evaluate import evaluate_random_policy
        cfg = _cfg(num_envs=8)
        result = evaluate_random_policy(cfg, n_episodes=10)
        assert "random_baseline_mean_return" in result
        assert isinstance(result["random_baseline_mean_return"], float)
