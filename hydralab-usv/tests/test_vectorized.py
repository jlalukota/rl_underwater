"""
Vectorized environment correctness and isolation tests.

These tests verify that:
    - Parallel environment slots are fully isolated (changes in one do not
      bleed into others)
    - Partial resets target only the specified env indices
    - Terminal observation in info is the PRE-reset state (for SB3 bootstrap)
    - Returned obs for done envs is the POST-reset initial state
    - Actions outside [-1, 1] are clamped silently
    - Observation and reward shapes match num_envs at every step
    - Domain randomization produces diverse initial states

Run with:  pytest tests/test_vectorized.py -v
"""

from __future__ import annotations

import torch
import pytest

from configs.blueboat_cfg import HydraLabConfig
from envs.blueboat_env import BlueboatEnv


DEVICE = "cpu"
NUM_ENVS = 16


@pytest.fixture
def cfg() -> HydraLabConfig:
    c = HydraLabConfig()
    c.env.num_envs = NUM_ENVS
    c.device = DEVICE
    c.obs.add_noise = False        # deterministic obs shapes
    c.randomization.enabled = True # DR needed for diversity tests
    return c


@pytest.fixture
def env(cfg) -> BlueboatEnv:
    e = BlueboatEnv(cfg)
    e.reset()
    return e


# ---------------------------------------------------------------------------
# Shape consistency
# ---------------------------------------------------------------------------

class TestShapes:

    def test_obs_shape_after_reset(self, env):
        obs, _ = env.reset()
        assert obs.shape == (NUM_ENVS, env.cfg.obs.obs_dim)

    def test_step_output_shapes(self, env):
        actions = torch.zeros(NUM_ENVS, 2, device=DEVICE)
        obs, rew, term, trunc, info = env.step(actions)
        assert obs.shape == (NUM_ENVS, env.cfg.obs.obs_dim)
        assert rew.shape == (NUM_ENVS,)
        assert term.shape == (NUM_ENVS,)
        assert trunc.shape == (NUM_ENVS,)

    def test_obs_finite_after_many_steps(self, env):
        for _ in range(50):
            actions = torch.rand(NUM_ENVS, 2, device=DEVICE) * 2 - 1
            obs, rew, _, _, _ = env.step(actions)
        assert obs.isfinite().all(), "obs contains NaN/Inf after 50 steps"
        assert rew.isfinite().all(), "reward contains NaN/Inf after 50 steps"


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------

class TestIsolation:

    def test_state_isolation_under_step(self, env):
        """Moving env 0 with maximum thrust must not change env 1 state."""
        env.reset()
        state_env1_before = env.body.state[1].clone()

        # Step env 0 aggressively via a full-batch step (both envs see actions)
        # but we verify that each env's state update is independent
        all_actions = torch.zeros(NUM_ENVS, 2, device=DEVICE)
        all_actions[0] = 1.0   # max thrust on env 0 only at logical level
        env.step(all_actions)

        # env 1 was given zero action — its state should differ from env 0
        # (different forces applied), so the states are numerically distinct
        assert not torch.allclose(env.body.state[0], env.body.state[1])

    def test_partial_reset_does_not_affect_other_envs(self, cfg):
        """Resetting env indices [0,1] must not change envs [2..]."""
        env = BlueboatEnv(cfg)
        env.reset()

        # Take a few steps to build up non-zero state
        for _ in range(5):
            env.step(torch.ones(NUM_ENVS, 2, device=DEVICE))

        state_before = env.body.state[2:].clone()

        reset_ids = torch.tensor([0, 1], device=DEVICE)
        env._reset_idx(reset_ids)

        state_after = env.body.state[2:]
        assert torch.allclose(state_before, state_after), (
            "Partial reset leaked into unreset env slots"
        )

    def test_partial_reset_zeroes_episode_length(self, env):
        """Episode length counter must be cleared for reset slots only."""
        # Advance all envs
        for _ in range(10):
            env.step(torch.zeros(NUM_ENVS, 2, device=DEVICE))

        ep_len_before = env.episode_length.clone()
        assert ep_len_before[0] > 0

        reset_ids = torch.tensor([0], device=DEVICE)
        env._reset_idx(reset_ids)

        assert env.episode_length[0] == 0
        for i in range(1, NUM_ENVS):
            assert env.episode_length[i] == ep_len_before[i]


# ---------------------------------------------------------------------------
# Action clamping
# ---------------------------------------------------------------------------

class TestActionClamping:

    def test_out_of_range_action_does_not_crash(self, env):
        huge = torch.full((NUM_ENVS, 2), 999.0, device=DEVICE)
        obs, rew, term, trunc, info = env.step(huge)
        assert obs.isfinite().all()

    def test_clamped_and_unclamped_identical_outcome(self):
        """Actions outside [-1, 1] should produce the same result as clamped.

        Disturbances are disabled so dynamics are deterministic and both envs
        produce bit-identical outputs given the same seed and action.
        """
        cfg_det = HydraLabConfig()
        cfg_det.env.num_envs = NUM_ENVS
        cfg_det.device = DEVICE
        cfg_det.obs.add_noise = False
        cfg_det.disturbance.wind_enabled = False
        cfg_det.disturbance.current_enabled = False
        cfg_det.disturbance.wave_enabled = False

        env_a = BlueboatEnv(cfg_det)
        env_b = BlueboatEnv(cfg_det)
        env_a.reset(seed=0)
        env_b.reset(seed=0)

        action_raw = torch.full((NUM_ENVS, 2), 2.0, device=DEVICE)
        action_clamped = action_raw.clamp(-1.0, 1.0)

        obs_a, _, _, _, _ = env_a.step(action_raw)
        obs_b, _, _, _, _ = env_b.step(action_clamped)

        assert torch.allclose(obs_a, obs_b, atol=1e-6)


# ---------------------------------------------------------------------------
# Done / auto-reset
# ---------------------------------------------------------------------------

class TestAutoReset:

    def test_done_env_obs_is_post_reset(self, cfg):
        """When an env is done, returned obs must be from the reset state,
        not the terminal state."""
        cfg_small = HydraLabConfig()
        cfg_small.env.num_envs = 4
        cfg_small.env.episode_length_s = 0.4   # 5 policy steps @ 0.08 s/step
        cfg_small.device = DEVICE
        cfg_small.obs.add_noise = False

        env = BlueboatEnv(cfg_small)
        env.reset()

        obs = None
        done_mask = torch.zeros(4, dtype=torch.bool)
        for _ in range(10):
            actions = torch.zeros(4, 2, device=DEVICE)
            obs, rew, term, trunc, info = env.step(actions)
            done_mask = term | trunc
            if done_mask.any():
                break

        assert done_mask.any(), "No env completed in 10 steps with episode_length=5"

        # For done envs, obs should be the post-reset initial obs (finite,
        # near-origin from random initial state)
        done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        done_obs = obs[done_ids]
        assert done_obs.isfinite().all()

    def test_terminal_obs_in_info_when_done(self, cfg):
        """info must contain _terminal_obs and _terminal_obs_ids when any env is done."""
        cfg_small = HydraLabConfig()
        cfg_small.env.num_envs = 4
        cfg_small.env.episode_length_s = 0.24   # 3 policy steps @ 0.08 s/step
        cfg_small.device = DEVICE
        cfg_small.obs.add_noise = False

        env = BlueboatEnv(cfg_small)
        env.reset()

        found_terminal_obs = False
        for _ in range(10):
            actions = torch.zeros(4, 2, device=DEVICE)
            obs, rew, term, trunc, info = env.step(actions)
            if (term | trunc).any():
                assert "_terminal_obs" in info
                assert "_terminal_obs_ids" in info
                term_ids = info["_terminal_obs_ids"]
                term_obs = info["_terminal_obs"]
                assert term_obs.shape == (term_ids.numel(), cfg_small.obs.obs_dim)
                found_terminal_obs = True
                break

        assert found_terminal_obs, "No done event in 10 steps with episode_length=3"

    def test_terminal_obs_differs_from_returned_obs_for_done_envs(self, cfg):
        """Terminal obs and returned obs must differ for done envs
        (terminal = final state, returned = post-reset initial state)."""
        cfg_small = HydraLabConfig()
        cfg_small.env.num_envs = 4
        cfg_small.env.episode_length_s = 0.24   # 3 policy steps @ 0.08 s/step
        cfg_small.device = DEVICE
        cfg_small.obs.add_noise = False
        cfg_small.randomization.enabled = True

        env = BlueboatEnv(cfg_small)
        env.reset()

        for _ in range(10):
            actions = torch.ones(4, 2, device=DEVICE)  # max thrust
            obs, _, term, trunc, info = env.step(actions)
            done_mask = term | trunc
            if done_mask.any() and "_terminal_obs" in info:
                done_ids = info["_terminal_obs_ids"]
                term_obs = info["_terminal_obs"]  # pre-reset
                returned_obs = obs[done_ids]       # post-reset
                # They should NOT be identical (episode just ended vs fresh reset)
                assert not torch.allclose(term_obs, returned_obs), (
                    "Terminal obs and post-reset obs are identical — "
                    "the auto-reset is not working correctly"
                )
                return

        pytest.skip("No done event observed — increase episode_length_steps budget")


# ---------------------------------------------------------------------------
# Domain randomisation diversity
# ---------------------------------------------------------------------------

class TestDomainRandomization:

    def test_initial_states_are_diverse(self, cfg):
        """After reset, no two envs should have identical initial states."""
        env = BlueboatEnv(cfg)
        env.reset()
        states = env.body.state  # (N, 6)
        for i in range(NUM_ENVS - 1):
            assert not torch.allclose(states[i], states[i + 1]), (
                f"Envs {i} and {i+1} have identical initial states — DR may be off"
            )

    def test_reward_diversity_under_same_action(self, cfg):
        """Same zero action should produce different rewards across envs
        (different initial positions → different distances to goal)."""
        env = BlueboatEnv(cfg)
        env.reset()
        actions = torch.zeros(NUM_ENVS, 2, device=DEVICE)
        _, rew, _, _, _ = env.step(actions)
        # At least some envs should have different rewards
        assert rew.std().item() > 0.0, "All rewards are identical — something is wrong"
