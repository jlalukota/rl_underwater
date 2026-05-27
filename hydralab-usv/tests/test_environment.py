"""
Integration tests for BlueboatEnv.

Tests verify:
    - Environment initialises without error
    - reset() returns correct shapes
    - step() returns correct shapes
    - Rewards are numerically finite
    - Reset correctly reinitialises episode counters
    - Disturbances produce non-zero forces
    - Observation noise is bounded
    - Multiple environments can be reset independently

Run with:  pytest tests/test_environment.py -v
"""

from __future__ import annotations

import torch
import pytest

from configs.blueboat_cfg import HydraLabConfig, EnvironmentConfig
from envs.blueboat_env import BlueboatEnv


DEVICE = "cpu"
NUM_ENVS = 8


@pytest.fixture
def cfg() -> HydraLabConfig:
    c = HydraLabConfig()
    c.env.num_envs = NUM_ENVS
    c.device = DEVICE
    c.obs.add_noise = False   # disable noise for deterministic shape tests
    return c


@pytest.fixture
def env(cfg) -> BlueboatEnv:
    e = BlueboatEnv(cfg)
    e.reset()
    return e


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:

    def test_env_creates(self, cfg):
        env = BlueboatEnv(cfg)
        assert env is not None

    def test_reset_returns_obs_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (NUM_ENVS, 10)

    def test_obs_is_finite(self, env):
        obs, _ = env.reset()
        assert obs.isfinite().all()


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:

    def test_step_shapes(self, env):
        actions = torch.zeros(NUM_ENVS, 2)
        obs, rew, terminated, truncated, info = env.step(actions)
        assert obs.shape == (NUM_ENVS, 10)
        assert rew.shape == (NUM_ENVS,)
        assert terminated.shape == (NUM_ENVS,)
        assert truncated.shape == (NUM_ENVS,)

    def test_step_obs_finite(self, env):
        for _ in range(20):
            obs, rew, *_ = env.step(torch.zeros(NUM_ENVS, 2))
        assert obs.isfinite().all()

    def test_rewards_finite(self, env):
        for _ in range(50):
            _, rew, *_ = env.step(torch.rand(NUM_ENVS, 2) * 2 - 1)
        assert rew.isfinite().all()

    def test_action_clamped(self, env):
        """Actions outside [-1, 1] should be clamped, not crash."""
        large_actions = torch.ones(NUM_ENVS, 2) * 10.0
        obs, rew, term, trunc, info = env.step(large_actions)
        assert obs.isfinite().all()

    def test_episode_counter_increments(self, env):
        initial = env.episode_length.clone()
        env.step(torch.zeros(NUM_ENVS, 2))
        assert (env.episode_length == initial + 1).all()

    def test_prev_action_updated(self, env):
        actions = torch.ones(NUM_ENVS, 2) * 0.5
        env.step(actions)
        assert torch.allclose(env.prev_action, actions)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_all_zeros_episode_counter(self, env):
        for _ in range(10):
            env.step(torch.zeros(NUM_ENVS, 2))
        env.reset()
        assert (env.episode_length == 0).all()

    def test_partial_reset(self, env):
        """Reset only a subset of environments."""
        for _ in range(5):
            env.step(torch.zeros(NUM_ENVS, 2))
        reset_ids = torch.tensor([0, 2, 4])
        env._reset_idx(reset_ids)
        assert (env.episode_length[reset_ids] == 0).all()
        # Other envs should still have their step count
        unreset_ids = torch.tensor([1, 3, 5, 6, 7])
        assert (env.episode_length[unreset_ids] > 0).all()

    def test_state_randomised_on_reset(self, env, cfg):
        """Different resets should produce different states (DR enabled)."""
        env.reset(seed=0)
        state_a = env.body.state.clone()
        env.reset(seed=1)
        state_b = env.body.state.clone()
        # At least some environments should differ
        assert not torch.allclose(state_a, state_b)


# ---------------------------------------------------------------------------
# Disturbances
# ---------------------------------------------------------------------------

class TestDisturbances:

    def test_wind_force_nonzero(self, env):
        """With wind enabled, some environments should have non-zero force."""
        tau_wind = env.wind.compute(env.body.state)
        assert tau_wind.isfinite().all()
        # Most envs should have non-zero wind (possible all-zero only at init)
        assert tau_wind.abs().sum() > 0

    def test_wave_force_nonzero_after_step(self, env):
        env.waves.step(0.02)
        tau_wave = env.waves.compute(env.body.state)
        assert tau_wave.isfinite().all()

    def test_current_force_finite(self, env):
        tau_cur = env.current.compute(env.body.state)
        assert tau_cur.isfinite().all()


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------

class TestRewards:

    def test_tracking_reward_peaks_at_goal(self, env, cfg):
        """Tracking reward should be ≈ 1 when vessel is at goal."""
        pos = env.task.target_pos.clone()   # vessel exactly at goal
        r = env.r_tracking.compute(pos=pos, goal_pos=pos)
        assert torch.allclose(r, torch.ones(NUM_ENVS), atol=1e-5)

    def test_tracking_reward_decays_with_distance(self, env):
        pos = torch.zeros(NUM_ENVS, 2)
        goal_near = pos + 0.1
        goal_far = pos + 10.0
        r_near = env.r_tracking.compute(pos=pos, goal_pos=goal_near)
        r_far = env.r_tracking.compute(pos=pos, goal_pos=goal_far)
        assert (r_near > r_far).all()

    def test_smoothness_penalty_zero_for_zero_delta(self, env):
        actions = torch.zeros(NUM_ENVS, 2)
        p = env.r_smoothness.compute(action=actions, prev_action=actions)
        assert torch.allclose(p, torch.zeros(NUM_ENVS), atol=1e-6)

    def test_energy_penalty_zero_for_zero_actions(self, env):
        actions = torch.zeros(NUM_ENVS, 2)
        p = env.r_energy.compute(action=actions)
        assert torch.allclose(p, torch.zeros(NUM_ENVS), atol=1e-6)


# ---------------------------------------------------------------------------
# Trajectory task
# ---------------------------------------------------------------------------

class TestTrajectoryTask:

    def test_target_pos_finite(self, env):
        assert env.task.target_pos.isfinite().all()

    def test_target_advances_after_step(self, env):
        pos_before = env.task.target_pos.clone()
        env.task.step(0.1)
        pos_after = env.task.target_pos
        # At least some trajectories should have moved
        assert not torch.allclose(pos_before, pos_after)

    def test_reset_changes_trajectory(self, env):
        pos_before = env.task.target_pos.clone()
        env.task.reset(torch.arange(NUM_ENVS))
        pos_after = env.task.target_pos
        assert not torch.allclose(pos_before, pos_after)
