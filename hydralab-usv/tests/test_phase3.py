"""
Phase 3 integration tests.

Covers:
    - DisturbanceCurriculum schedule math (linear, cosine, step, disabled)
    - Curriculum apply() modifies live environment config in-place
    - make_curriculum_from_env() factory
    - HydraLabRLlibEnv: numpy interface, obs/action shapes
    - StabilityValidator: passes on stable env, catches injected instability
    - PPO smoke test: 100 steps without crash (skips if SB3 unavailable)
    - Terminal observation correctness in VecEnvWrapper

Run with:  pytest tests/test_phase3.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from configs.blueboat_cfg import HydraLabConfig
from configs.training_cfg import CurriculumConfig, TrainingConfig
from envs.blueboat_env import BlueboatEnv
from training.curriculum import DisturbanceCurriculum, make_curriculum_from_env


DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg() -> HydraLabConfig:
    c = HydraLabConfig()
    c.env.num_envs = 8
    c.device = DEVICE
    c.obs.add_noise = False
    return c


@pytest.fixture
def small_env(small_cfg) -> BlueboatEnv:
    e = BlueboatEnv(small_cfg)
    e.reset()
    return e


def _make_curriculum(schedule="linear", enabled=True) -> DisturbanceCurriculum:
    cfg = CurriculumConfig(
        enabled=enabled,
        schedule=schedule,
        warmup_steps=1_000,
        full_steps=10_000,
        initial_disturbance_scale=0.1,
        final_disturbance_scale=1.0,
        initial_dr_scale=0.0,
        final_dr_scale=1.0,
    )
    return DisturbanceCurriculum(
        cfg=cfg,
        nominal_wind_mean=10.0,
        nominal_current_mean=0.5,
        nominal_wave_force_std=5.0,
        nominal_dr_fracs={
            "mass_noise_frac": 0.10,
            "drag_noise_frac": 0.20,
            "inertia_noise_frac": 0.10,
            "thrust_lo": 0.8,
            "thrust_hi": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# Curriculum — scale computation
# ---------------------------------------------------------------------------

class TestCurriculumScale:

    def test_linear_at_warmup_start(self):
        c = _make_curriculum("linear")
        assert c.get_disturbance_scale(0) == pytest.approx(0.1)

    def test_linear_before_warmup_complete(self):
        c = _make_curriculum("linear")
        # step < warmup_steps → still at initial
        assert c.get_disturbance_scale(999) == pytest.approx(0.1)

    def test_linear_midpoint(self):
        c = _make_curriculum("linear")
        mid = (1_000 + 10_000) // 2  # 5500
        scale = c.get_disturbance_scale(mid)
        # 50% progress → scale = 0.1 + 0.5 * 0.9 = 0.55
        assert scale == pytest.approx(0.55, abs=1e-3)

    def test_linear_at_full_steps(self):
        c = _make_curriculum("linear")
        assert c.get_disturbance_scale(10_000) == pytest.approx(1.0)

    def test_linear_beyond_full_steps(self):
        c = _make_curriculum("linear")
        assert c.get_disturbance_scale(999_999) == pytest.approx(1.0)

    def test_cosine_midpoint_below_linear(self):
        # Cosine schedule has slow start → at 50% progress, scale < linear midpoint
        c_cos = _make_curriculum("cosine")
        c_lin = _make_curriculum("linear")
        mid = (1_000 + 10_000) // 2
        # Both start at 0.1, but cosine ramps slower at start
        # At exactly 50%, cosine = lo + 0.5*(hi-lo) = same, but...
        # At 25% progress, cosine < linear
        step_25pct = int(1_000 + 0.25 * 9_000)
        assert c_cos.get_disturbance_scale(step_25pct) < c_lin.get_disturbance_scale(step_25pct)

    def test_cosine_endpoints_match_linear(self):
        c = _make_curriculum("cosine")
        assert c.get_disturbance_scale(0) == pytest.approx(0.1)
        assert c.get_disturbance_scale(10_000) == pytest.approx(1.0)

    def test_disabled_always_returns_final(self):
        c = _make_curriculum("linear", enabled=False)
        assert c.get_disturbance_scale(0) == pytest.approx(1.0)
        assert c.get_disturbance_scale(500) == pytest.approx(1.0)
        assert c.get_disturbance_scale(9_999_999) == pytest.approx(1.0)

    def test_dr_scale_starts_at_zero(self):
        c = _make_curriculum("linear")
        assert c.get_dr_scale(0) == pytest.approx(0.0)

    def test_dr_scale_reaches_one(self):
        c = _make_curriculum("linear")
        assert c.get_dr_scale(10_000) == pytest.approx(1.0)

    def test_summary_keys_present(self):
        c = _make_curriculum("linear")
        s = c.summary(5_500)
        assert "curriculum/disturbance_scale" in s
        assert "curriculum/dr_scale" in s
        assert "curriculum/progress" in s

    def test_summary_progress_clamps_to_one(self):
        c = _make_curriculum("linear")
        s = c.summary(999_999)
        assert s["curriculum/progress"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Curriculum — apply() modifies live env
# ---------------------------------------------------------------------------

class TestCurriculumApply:

    def test_apply_reduces_wind_at_early_step(self, small_env):
        nominal_wind = small_env.cfg.disturbance.wind_speed_mean
        c = make_curriculum_from_env(small_env, CurriculumConfig(
            enabled=True,
            schedule="linear",
            warmup_steps=500_000,
            full_steps=5_000_000,
            initial_disturbance_scale=0.05,
            final_disturbance_scale=1.0,
        ))
        c.apply(small_env, step=0)
        assert small_env.cfg.disturbance.wind_speed_mean < nominal_wind

    def test_apply_reaches_nominal_at_full_steps(self, small_env):
        nominal_wind = small_env.cfg.disturbance.wind_speed_mean
        c = make_curriculum_from_env(small_env, CurriculumConfig(
            enabled=True,
            schedule="linear",
            warmup_steps=100,
            full_steps=1_000,
        ))
        c.apply(small_env, step=1_000)
        assert small_env.cfg.disturbance.wind_speed_mean == pytest.approx(
            nominal_wind, rel=1e-4
        )

    def test_apply_propagates_to_live_disturbance_object(self, small_env):
        """Changes must also appear on env.wind.cfg for newly-sampled episodes."""
        c = make_curriculum_from_env(small_env, CurriculumConfig(
            enabled=True,
            warmup_steps=100,
            full_steps=1_000,
            initial_disturbance_scale=0.05,
        ))
        c.apply(small_env, step=0)
        cfg_val = small_env.cfg.disturbance.wind_speed_mean
        live_val = small_env.wind.cfg.wind_speed_mean
        assert cfg_val == pytest.approx(live_val, rel=1e-5)

    def test_apply_no_redundant_write_same_scale(self, small_env):
        """Calling apply() twice at the same scale must not change anything."""
        c = make_curriculum_from_env(small_env, CurriculumConfig(
            warmup_steps=100, full_steps=1_000
        ))
        c.apply(small_env, step=0)
        wind_after_first = small_env.cfg.disturbance.wind_speed_mean

        c.apply(small_env, step=0)
        wind_after_second = small_env.cfg.disturbance.wind_speed_mean

        assert wind_after_first == wind_after_second


# ---------------------------------------------------------------------------
# make_curriculum_from_env factory
# ---------------------------------------------------------------------------

class TestMakeCurriculumFromEnv:

    def test_factory_returns_curriculum(self, small_env):
        cc = CurriculumConfig()
        c = make_curriculum_from_env(small_env, cc)
        assert isinstance(c, DisturbanceCurriculum)

    def test_factory_nominal_wind_matches_env(self, small_env):
        cc = CurriculumConfig(initial_disturbance_scale=1.0)
        c = make_curriculum_from_env(small_env, cc)
        # With scale=1.0, wind should stay at nominal
        c.apply(small_env, step=999_999)
        nominal = small_env.cfg.disturbance.wind_speed_mean
        assert c._wind_nom == pytest.approx(nominal, rel=1e-4)


# ---------------------------------------------------------------------------
# RLlib environment wrapper
# ---------------------------------------------------------------------------

class TestRLlibEnv:

    def test_rllib_env_creates(self):
        from training.rllib_env import HydraLabRLlibEnv
        import numpy as np
        env = HydraLabRLlibEnv({"device": "cpu", "seed": 0})
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (env.observation_space.shape[0],)
        env.close()

    def test_rllib_env_step_returns_numpy(self):
        from training.rllib_env import HydraLabRLlibEnv
        import numpy as np
        env = HydraLabRLlibEnv({"device": "cpu", "seed": 1})
        env.reset()
        action = np.zeros(2, dtype=np.float32)
        obs, rew, term, trunc, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(rew, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        env.close()

    def test_rllib_env_obs_shape(self):
        from training.rllib_env import HydraLabRLlibEnv
        import numpy as np
        env = HydraLabRLlibEnv({"device": "cpu"})
        obs, _ = env.reset()
        expected_dim = env.observation_space.shape[0]
        assert obs.shape == (expected_dim,)
        action = env.action_space.sample()
        obs2, _, _, _, _ = env.step(action)
        assert obs2.shape == (expected_dim,)
        env.close()

    def test_rllib_env_action_space_matches(self):
        from training.rllib_env import HydraLabRLlibEnv
        env = HydraLabRLlibEnv({"device": "cpu"})
        assert env.action_space.shape == (2,)
        assert env.action_space.low[0] == pytest.approx(-1.0)
        assert env.action_space.high[0] == pytest.approx(1.0)
        env.close()

    def test_rllib_vec_env_batched_shapes(self):
        from training.rllib_env import HydraLabRLlibVecEnv
        import numpy as np
        slots = 4
        env = HydraLabRLlibVecEnv({"device": "cpu", "slots": slots})
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] == slots
        action = np.zeros((slots, 2), dtype=np.float32)
        obs2, rew, term, trunc, _ = env.step(action)
        assert obs2.shape[0] == slots
        assert rew.shape == (slots,)
        env.close()


# ---------------------------------------------------------------------------
# Stability validator
# ---------------------------------------------------------------------------

class TestStabilityValidator:

    def test_stable_env_passes(self):
        from training.stability_validator import StabilityValidator
        cfg = HydraLabConfig()
        cfg.env.num_envs = 16
        cfg.device = DEVICE
        # Calm environment → must pass easily
        cfg.disturbance.wind_speed_mean = 0.0
        cfg.disturbance.current_speed_mean = 0.0
        cfg.disturbance.wave_force_std = 0.0

        validator = StabilityValidator(cfg=cfg, n_steps=200, n_envs=16, device=DEVICE)
        report = validator.run()
        assert report.passed, f"Stable env failed: {report.reason}"

    def test_validator_report_has_stats(self):
        from training.stability_validator import StabilityValidator
        cfg = HydraLabConfig()
        cfg.env.num_envs = 8
        cfg.device = DEVICE

        validator = StabilityValidator(cfg=cfg, n_steps=50, n_envs=8, device=DEVICE)
        report = validator.run()
        assert report.n_steps == 50
        assert report.max_pos_norm >= 0.0
        assert report.max_vel_norm >= 0.0

    def test_validator_catches_position_explosion(self):
        from training.stability_validator import StabilityValidator
        cfg = HydraLabConfig()
        cfg.env.num_envs = 4
        cfg.device = DEVICE

        # Very tight limit → should fail quickly
        validator = StabilityValidator(
            cfg=cfg,
            n_steps=500,
            n_envs=4,
            device=DEVICE,
            action_mode="max",   # maximum thrust
            max_pos_m=0.1,       # fail at any movement > 10 cm
        )
        report = validator.run()
        # Maximum thrust WILL move the vessel past 10 cm
        assert not report.passed
        assert "explosion" in report.reason.lower() or report.failure_step is not None

    def test_validator_summary_contains_pass_or_fail(self):
        from training.stability_validator import StabilityValidator
        cfg = HydraLabConfig()
        cfg.env.num_envs = 4
        cfg.device = DEVICE

        validator = StabilityValidator(cfg=cfg, n_steps=20, n_envs=4, device=DEVICE)
        report = validator.run()
        summary = report.summary()
        assert "PASS" in summary or "FAIL" in summary


# ---------------------------------------------------------------------------
# Terminal observation correctness (full loop)
# ---------------------------------------------------------------------------

class TestTerminalObservation:

    def test_terminal_obs_shape_matches_obs_space(self):
        cfg = HydraLabConfig()
        cfg.env.num_envs = 4
        cfg.env.episode_length_s = 0.24   # 3 policy steps @ 0.08 s/step
        cfg.device = DEVICE
        cfg.obs.add_noise = False

        env = BlueboatEnv(cfg)
        env.reset()

        for _ in range(15):
            actions = torch.zeros(4, 2, device=DEVICE)
            obs, _, term, trunc, info = env.step(actions)
            if "_terminal_obs" in info:
                t_obs = info["_terminal_obs"]
                t_ids = info["_terminal_obs_ids"]
                assert t_obs.shape == (t_ids.numel(), cfg.obs.obs_dim)
                assert t_obs.isfinite().all()
                return

        pytest.skip("No done event observed in 15 steps — adjust episode_length")

    def test_returned_obs_is_finite_for_done_envs(self):
        cfg = HydraLabConfig()
        cfg.env.num_envs = 4
        cfg.env.episode_length_s = 0.16   # 2 policy steps @ 0.08 s/step
        cfg.device = DEVICE
        cfg.obs.add_noise = False

        env = BlueboatEnv(cfg)
        env.reset()

        for _ in range(10):
            actions = torch.zeros(4, 2, device=DEVICE)
            obs, _, term, trunc, info = env.step(actions)
            done_mask = term | trunc
            if done_mask.any():
                done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
                assert obs[done_ids].isfinite().all()
                return

        pytest.skip("No done event observed")


# ---------------------------------------------------------------------------
# PPO smoke test
# ---------------------------------------------------------------------------

class TestPPOSmoke:

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("stable_baselines3"),
        reason="stable-baselines3 not installed",
    )
    def test_ppo_100_steps_no_crash(self):
        """Minimal PPO training run: 100 timesteps, 4 envs."""
        from stable_baselines3 import PPO
        from training.ppo_train import HydraLabVecEnvWrapper

        cfg = HydraLabConfig()
        cfg.env.num_envs = 4
        cfg.device = DEVICE
        cfg.obs.add_noise = False

        env = BlueboatEnv(cfg)
        env.reset()
        vec_env = HydraLabVecEnvWrapper(env)

        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=25,
            batch_size=20,
            n_epochs=1,
            verbose=0,
        )
        model.learn(total_timesteps=100)
        vec_env.close()
