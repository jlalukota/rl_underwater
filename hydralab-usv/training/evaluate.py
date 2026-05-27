"""
Post-training policy evaluation script.

Loads a saved SB3 PPO checkpoint and runs evaluation episodes across
multiple parallel environments.  Reports tracking error, episode return,
and goal-capture statistics.

Usage:
    python training/evaluate.py --checkpoint checkpoints/ppo_blueboat_final
    python training/evaluate.py --checkpoint path/to/model --n-episodes 200
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

import torch

from configs.blueboat_cfg import HydraLabConfig
from envs.blueboat_env import BlueboatEnv
from envs.episode_stats import EpisodeStats

try:
    from stable_baselines3 import PPO
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


def evaluate_policy(
    model_path: str,
    cfg: Optional[HydraLabConfig] = None,
    n_episodes: int = 100,
    render: bool = False,
    verbose: bool = True,
) -> dict:
    """Evaluate a saved PPO policy.

    Args:
        model_path:  Path to saved SB3 PPO model (without .zip).
        cfg:         Environment config; defaults to HydraLabConfig().
        n_episodes:  Target number of episodes to collect.
        render:      Whether to call the debug renderer each step.
        verbose:     Print progress.

    Returns:
        Dict of aggregate statistics.
    """
    if not _SB3_AVAILABLE:
        raise ImportError("stable-baselines3 required: pip install stable-baselines3")

    cfg = cfg or HydraLabConfig()
    # Disable noise during evaluation for cleaner metrics
    cfg.obs.add_noise = False

    env = BlueboatEnv(cfg)
    obs, _ = env.reset(seed=99)

    model = PPO.load(model_path, device=cfg.device)

    stats = EpisodeStats(cfg.env.num_envs, cfg.device)

    if render:
        from visualization.debug_renderer import DebugRenderer
        renderer = DebugRenderer(num_envs_to_show=min(4, cfg.env.num_envs))

    episode_count = 0
    returns: list[float] = []
    lengths: list[int] = []
    tracking_errors: list[float] = []

    t0 = time.perf_counter()

    while episode_count < n_episodes:
        actions_np, _ = model.predict(obs.cpu().numpy(), deterministic=True)
        actions = torch.from_numpy(actions_np).to(cfg.device)

        obs, rewards, terminated, truncated, info = env.step(actions)

        dist_to_goal = env.r_tracking.distance(env.body.pos, env.task.target_pos)
        done_mask = terminated | truncated

        stats.update(rewards, dist_to_goal)
        summaries = stats.collect_done(done_mask)
        done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            stats.reset(done_ids)

        for s in summaries:
            returns.append(s["episode_return"])
            lengths.append(s["episode_length"])
            tracking_errors.append(s["mean_tracking_error"])
            episode_count += 1

        if render:
            renderer.update(env, rewards)
            if episode_count % 10 == 0:
                renderer.render(env)

    elapsed = time.perf_counter() - t0

    result = {
        "n_episodes": episode_count,
        "mean_return": statistics.mean(returns),
        "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "mean_length": statistics.mean(lengths),
        "mean_tracking_error_m": statistics.mean(tracking_errors),
        "eval_time_s": elapsed,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("  HydraLab-USV Evaluation Report")
        print("=" * 50)
        print(f"  Episodes evaluated  : {result['n_episodes']}")
        print(f"  Mean return         : {result['mean_return']:.2f} ± {result['std_return']:.2f}")
        print(f"  Mean episode length : {result['mean_length']:.1f} steps")
        print(f"  Mean tracking error : {result['mean_tracking_error_m']:.3f} m")
        print(f"  Eval time           : {result['eval_time_s']:.1f} s")
        print("=" * 50 + "\n")

    env.close()
    return result


def evaluate_random_policy(
    cfg: Optional[HydraLabConfig] = None,
    n_episodes: int = 50,
) -> dict:
    """Evaluate a random baseline policy (for sanity checking environment).

    Args:
        cfg:        Environment config.
        n_episodes: Number of episodes.

    Returns:
        Aggregate statistics dict.
    """
    cfg = cfg or HydraLabConfig()
    cfg.obs.add_noise = False
    env = BlueboatEnv(cfg)
    obs, _ = env.reset(seed=42)
    stats = EpisodeStats(cfg.env.num_envs, cfg.device)
    returns: list[float] = []
    episode_count = 0

    while episode_count < n_episodes:
        actions = torch.rand(cfg.env.num_envs, 2, device=cfg.device) * 2.0 - 1.0
        obs, rewards, terminated, truncated, info = env.step(actions)
        dist_to_goal = env.r_tracking.distance(env.body.pos, env.task.target_pos)
        done_mask = terminated | truncated
        stats.update(rewards, dist_to_goal)
        summaries = stats.collect_done(done_mask)
        done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            stats.reset(done_ids)
        for s in summaries:
            returns.append(s["episode_return"])
            episode_count += 1

    env.close()
    result = stats.summary()
    result["random_baseline_mean_return"] = (
        sum(returns) / max(len(returns), 1)
    )
    print(f"[Random baseline] Mean return: {result['random_baseline_mean_return']:.2f}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate HydraLab-USV policy")
    p.add_argument("--checkpoint", required=True, help="Path to SB3 PPO model")
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--render", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = HydraLabConfig()
    cfg.device = args.device
    cfg.env.num_envs = 32  # smaller for eval
    evaluate_policy(
        args.checkpoint,
        cfg=cfg,
        n_episodes=args.n_episodes,
        render=args.render,
    )
