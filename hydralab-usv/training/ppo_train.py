"""
PPO training script for HydraLab-USV.

Integrates with Stable-Baselines3 via a VecEnv wrapper.
The wrapper bridges the batched BlueboatEnv tensor API to
the SB3 VecEnv numpy interface.

Usage:
    python training/ppo_train.py
    python training/ppo_train.py --num-envs 2048 --total-steps 50_000_000
    python training/ppo_train.py --config training/hyperparams/ppo_curriculum.yaml

Architecture:
    BlueboatEnv (GPU tensors)
        → HydraLabVecEnvWrapper (numpy/torch bridge)
            → VecMonitor
                → SB3 PPO (policy gradient)
                    + HydraLabMetricsCallback
                    + CurriculumCallback (optional)
                    + BestModelCallback
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

from configs.blueboat_cfg import HydraLabConfig
from configs.training_cfg import TrainingConfig, CurriculumConfig
from envs.blueboat_env import BlueboatEnv
from tasks.trajectory_tracking import TrajectoryType


# ---------------------------------------------------------------------------
# VecEnv wrapper bridging tensor API to SB3
# ---------------------------------------------------------------------------

if _SB3_AVAILABLE:

    class HydraLabVecEnvWrapper(VecEnv):
        """Wraps BlueboatEnv to present a SB3-compatible VecEnv interface.

        Converts PyTorch tensors ↔ numpy arrays at the interface boundary.
        All heavy computation stays on GPU.

        Terminal observation handling:
            BlueboatEnv.step() returns POST-RESET obs for done environments
            and stores the PRE-RESET terminal obs in
            info["_terminal_obs"] / info["_terminal_obs_ids"].
            This wrapper moves those into the per-env info dicts as
            info[i]["terminal_observation"] for SB3 PPO value bootstrap.
        """

        def __init__(self, env: BlueboatEnv) -> None:
            self.env = env
            N = env.num_envs
            obs_space = env.observation_space
            act_space = env.action_space
            super().__init__(N, obs_space, act_space)

            self._obs_buf = np.zeros((N, obs_space.shape[0]), dtype=np.float32)
            self._rew_buf = np.zeros(N, dtype=np.float32)
            self._done_buf = np.zeros(N, dtype=bool)
            self._info_buf: list[dict] = [{} for _ in range(N)]

        # ------------------------------------------------------------------
        # VecEnv interface
        # ------------------------------------------------------------------

        def reset(self) -> np.ndarray:
            obs, _ = self.env.reset()
            self._obs_buf[:] = obs.cpu().numpy()
            return self._obs_buf.copy()

        def step_async(self, actions: np.ndarray) -> None:
            self._pending_actions = torch.from_numpy(actions).to(self.env.device)

        def step_wait(self):
            obs, rew, terminated, truncated, info = self.env.step(
                self._pending_actions
            )
            dones = (terminated | truncated).cpu().numpy()
            self._obs_buf[:] = obs.cpu().numpy()
            self._rew_buf[:] = rew.cpu().numpy()
            self._done_buf[:] = dones

            # Reset per-env info dicts
            for i in range(self.num_envs):
                self._info_buf[i] = {}

            # Populate terminal_observation for SB3 value bootstrap.
            # BlueboatEnv stores pre-reset obs under "_terminal_obs" /
            # "_terminal_obs_ids" (set only when done_ids is non-empty).
            terminal_ids = info.get("_terminal_obs_ids")
            terminal_obs = info.get("_terminal_obs")
            if terminal_ids is not None and terminal_obs is not None:
                term_np = terminal_obs.cpu().numpy()
                for j, env_id in enumerate(terminal_ids.tolist()):
                    self._info_buf[env_id]["terminal_observation"] = term_np[j]

            return (
                self._obs_buf.copy(),
                self._rew_buf.copy(),
                self._done_buf.copy(),
                list(self._info_buf),
            )

        def close(self) -> None:
            self.env.close()

        def get_attr(self, attr_name, indices=None):
            return [getattr(self.env, attr_name)] * self.num_envs

        def set_attr(self, attr_name, value, indices=None):
            setattr(self.env, attr_name, value)

        def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
            return [getattr(self.env, method_name)(*method_args, **method_kwargs)]

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * self.num_envs

        def get_images(self):
            return [None] * self.num_envs

        def seed(self, seed=None):
            return [None] * self.num_envs


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_vec_env(
    cfg: Optional[HydraLabConfig] = None,
    traj_type: TrajectoryType = TrajectoryType.SINUSOIDAL,
) -> "HydraLabVecEnvWrapper":
    """Construct a vectorised training environment."""
    if not _SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required for this function.")
    cfg = cfg or HydraLabConfig()
    env = BlueboatEnv(cfg, traj_type=traj_type)
    env.reset()
    return HydraLabVecEnvWrapper(env)


def load_training_config(path: str) -> TrainingConfig:
    """Load TrainingConfig from a YAML file."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to load training configs.") from e

    with open(path) as f:
        data = yaml.safe_load(f)

    tc = TrainingConfig()
    for k, v in data.items():
        if k == "policy":
            from configs.training_cfg import PolicyConfig
            pc = PolicyConfig()
            for pk, pv in v.items():
                setattr(pc, pk, pv)
            tc.policy = pc
        elif k == "curriculum":
            cc = CurriculumConfig()
            for ck, cv in v.items():
                setattr(cc, ck, cv)
            tc.curriculum = cc
        elif hasattr(tc, k):
            setattr(tc, k, v)
    return tc


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_ppo(
    env_cfg: Optional[HydraLabConfig] = None,
    train_cfg: Optional[TrainingConfig] = None,
    traj_type: TrajectoryType = TrajectoryType.SINUSOIDAL,
    verbose: int = 1,
) -> None:
    """Train a PPO policy on HydraLabUSV trajectory tracking.

    Args:
        env_cfg:    HydraLabConfig for the environment.  Uses default if None.
        train_cfg:  TrainingConfig with PPO hyperparameters.  Uses default if None.
        traj_type:  Trajectory type for the tracking task.
        verbose:    SB3 verbosity level.
    """
    if not _SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is required.  Install with:\n"
            "  pip install stable-baselines3"
        )

    env_cfg = env_cfg or HydraLabConfig()
    train_cfg = train_cfg or TrainingConfig()
    tc = train_cfg

    print(f"[HydraLab] Starting PPO training — {env_cfg.env.num_envs} parallel environments")
    print(f"[HydraLab] Device: {env_cfg.device}")
    print(f"[HydraLab] Curriculum: {'enabled' if tc.curriculum.enabled else 'disabled'}")

    Path(tc.log_dir).mkdir(parents=True, exist_ok=True)
    Path(tc.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(env_cfg, traj_type)
    vec_env = VecMonitor(vec_env, tc.log_dir)

    # Activation function
    _act_map = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "elu": torch.nn.ELU}
    act_fn = _act_map.get(tc.policy.activation, torch.nn.Tanh)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=tc.learning_rate,
        n_steps=tc.n_steps,
        batch_size=tc.batch_size,
        n_epochs=tc.n_epochs,
        gamma=tc.gamma,
        gae_lambda=tc.gae_lambda,
        clip_range=tc.clip_range,
        clip_range_vf=tc.clip_range_vf,
        ent_coef=tc.ent_coef,
        vf_coef=tc.vf_coef,
        max_grad_norm=tc.max_grad_norm,
        normalize_advantage=tc.normalize_advantage,
        seed=tc.seed,
        verbose=verbose,
        tensorboard_log=tc.log_dir,
        device=env_cfg.device,
        policy_kwargs=dict(
            net_arch=dict(
                pi=tc.policy.net_arch,
                vf=tc.policy.net_arch,
            ),
            activation_fn=act_fn,
        ),
    )

    # --- Callbacks ---
    from training.callbacks import (
        HydraLabMetricsCallback,
        CurriculumCallback,
        BestModelCallback,
    )

    # Unwrapped BlueboatEnv needed by metric/curriculum callbacks
    unwrapped_env: BlueboatEnv = vec_env.unwrapped.env  # type: ignore[attr-defined]

    callbacks = [
        CheckpointCallback(
            save_freq=max(tc.checkpoint_freq // env_cfg.env.num_envs, 1),
            save_path=tc.checkpoint_dir,
            name_prefix="ppo_blueboat",
            verbose=1,
        ),
        HydraLabMetricsCallback(unwrapped_env, log_freq=1, verbose=0),
        BestModelCallback(save_path=tc.checkpoint_dir, check_freq=10_000, verbose=1),
    ]

    if tc.curriculum.enabled:
        from training.curriculum import make_curriculum_from_env
        curriculum = make_curriculum_from_env(unwrapped_env, tc.curriculum)
        callbacks.append(
            CurriculumCallback(
                curriculum=curriculum,
                env=unwrapped_env,
                update_freq=max(tc.n_steps * env_cfg.env.num_envs // 4, 256),
                verbose=0,
            )
        )

    t0 = time.time()
    model.learn(
        total_timesteps=tc.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - t0

    model.save(f"{tc.checkpoint_dir}/ppo_blueboat_final")
    print(f"[HydraLab] Training complete in {elapsed:.1f}s")
    print(f"[HydraLab] SPS: {tc.total_timesteps / elapsed:.0f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HydraLab-USV PPO Training")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML training config (overrides other flags if set)")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--total-steps", type=int, default=50_000_000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-dir", type=str, default="logs/ppo_blueboat")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--traj-type", type=str, default="sinusoidal",
                   choices=[t.value for t in TrajectoryType])
    p.add_argument("--curriculum", action="store_true",
                   help="Enable disturbance curriculum")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    env_cfg = HydraLabConfig()
    env_cfg.env.num_envs = args.num_envs
    env_cfg.device = args.device

    if args.config:
        train_cfg = load_training_config(args.config)
    else:
        train_cfg = TrainingConfig(
            total_timesteps=args.total_steps,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
        )
        if args.curriculum:
            train_cfg.curriculum.enabled = True

    traj_type = TrajectoryType(args.traj_type)

    train_ppo(env_cfg=env_cfg, train_cfg=train_cfg, traj_type=traj_type)
