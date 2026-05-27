"""
RLlib-compatible environment wrapper.

RLlib manages its own worker-level parallelism and creates many copies of the
environment across CPUs/GPUs.  Each copy must be a standard gymnasium.Env
returning numpy arrays (not tensors).

HydraLabRLlibEnv wraps a single BlueboatEnv slot (num_envs=1), unbatches
tensors to plain numpy, and satisfies the RLlib single-agent interface.

For large-scale RLlib training, use the vectorised variant
HydraLabRLlibVecEnv which exposes multiple slots as a batched env.

RLlib registration:
    from ray.tune import register_env
    register_env("HydraLabUSV-v0", lambda cfg: HydraLabRLlibEnv(cfg))

    trainer = PPO(env="HydraLabUSV-v0", config={
        "env_config": {"num_envs": 1, "device": "cpu"},
        ...
    })

TODO: Add multi-agent variant for formation-control experiments.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

from configs.blueboat_cfg import HydraLabConfig
from tasks.trajectory_tracking import TrajectoryType


class HydraLabRLlibEnv:
    """Single-agent RLlib gymnasium wrapper around BlueboatEnv.

    Wraps one slot of a BlueboatEnv(num_envs=1) and converts tensors to numpy.
    Satisfies the gymnasium.Env interface expected by RLlib.

    env_config keys (all optional):
        cfg            HydraLabConfig instance
        device         "cpu" or "cuda"
        traj_type      TrajectoryType enum value or string
        seed           random seed
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: Optional[dict[str, Any]] = None) -> None:
        env_config = env_config or {}

        cfg: HydraLabConfig = env_config.get("cfg", HydraLabConfig())
        cfg.env.num_envs = 1            # single env slot per RLlib worker
        cfg.device = env_config.get("device", "cpu")
        cfg.seed = env_config.get("seed", 42)

        traj_raw = env_config.get("traj_type", TrajectoryType.SINUSOIDAL)
        if isinstance(traj_raw, str):
            traj_raw = TrajectoryType(traj_raw)

        # Import here to avoid circular import at module load
        from envs.blueboat_env import BlueboatEnv
        self._env = BlueboatEnv(cfg, traj_type=traj_raw)

        oc = cfg.obs
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(oc.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        obs, info = self._env.reset(seed=seed)
        return obs[0].cpu().numpy(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        import torch
        action_t = torch.from_numpy(action.astype(np.float32)).unsqueeze(0)
        obs, rew, terminated, truncated, info = self._env.step(action_t)
        return (
            obs[0].cpu().numpy(),
            float(rew[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            info,
        )

    def close(self) -> None:
        self._env.close()

    def render(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Vectorised RLlib variant (exposes K slots per worker for higher throughput)
# ---------------------------------------------------------------------------

class HydraLabRLlibVecEnv(HydraLabRLlibEnv):
    """Multi-slot RLlib env for higher per-worker throughput.

    Each RLlib worker creates one of these; it manages K environments
    internally via GPU batch operations rather than N separate workers.
    Returns batched numpy arrays of shape (K, ...).

    Note: RLlib requires the env to expose single-agent obs/action spaces
    even if it internally batches.  This variant is designed for use with
    RLlib's `num_envs_per_worker` setting.

    env_config keys (in addition to HydraLabRLlibEnv):
        slots  int — number of environment slots per worker (default 16)
    """

    def __init__(self, env_config: Optional[dict[str, Any]] = None) -> None:
        env_config = env_config or {}
        slots = env_config.get("slots", 16)

        cfg: HydraLabConfig = env_config.get("cfg", HydraLabConfig())
        cfg.env.num_envs = slots
        cfg.device = env_config.get("device", "cpu")

        traj_raw = env_config.get("traj_type", TrajectoryType.SINUSOIDAL)
        if isinstance(traj_raw, str):
            traj_raw = TrajectoryType(traj_raw)

        from envs.blueboat_env import BlueboatEnv
        self._env = BlueboatEnv(cfg, traj_type=traj_raw)
        self._slots = slots

        oc = cfg.obs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(slots, oc.obs_dim), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(slots, 2), dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed)
        return obs.cpu().numpy(), info

    def step(self, action: np.ndarray):
        import torch
        action_t = torch.from_numpy(action.astype(np.float32))
        obs, rew, term, trunc, info = self._env.step(action_t)
        return (
            obs.cpu().numpy(),
            rew.cpu().numpy(),
            term.cpu().numpy(),
            trunc.cpu().numpy(),
            info,
        )
