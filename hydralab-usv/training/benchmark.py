"""
Throughput benchmark for HydraLab-USV.

Measures simulated steps per second (SPS) for a range of num_envs values,
validating Phase 3 parallelisation targets:
    - Target: 1000+ envs
    - Stretch: 5000+ envs

Usage:
    cd hydralab-usv
    python training/benchmark.py
    python training/benchmark.py --device cuda --max-envs 5000
"""

from __future__ import annotations

import argparse
import time

import torch

from configs.blueboat_cfg import HydraLabConfig
from envs.blueboat_env import BlueboatEnv


def benchmark(
    num_envs: int,
    device: str,
    warmup_steps: int = 50,
    measure_steps: int = 500,
) -> dict:
    """Run a timed rollout and return throughput metrics.

    Args:
        num_envs:       Number of parallel environments.
        device:         Torch device ("cuda" or "cpu").
        warmup_steps:   Steps to discard before timing (JIT / cache warmup).
        measure_steps:  Steps to time.

    Returns:
        dict with keys: num_envs, sps (steps/sec), ms_per_step, device.
    """
    cfg = HydraLabConfig()
    cfg.env.num_envs = num_envs
    cfg.device = device
    # Disable noise for pure dynamics throughput measurement
    cfg.obs.add_noise = False

    env = BlueboatEnv(cfg)
    env.reset(seed=0)

    actions = torch.rand(num_envs, 2, device=device) * 2.0 - 1.0

    # Warmup
    for _ in range(warmup_steps):
        env.step(actions)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed measurement
    t0 = time.perf_counter()
    for _ in range(measure_steps):
        env.step(actions)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    env.close()

    total_steps = num_envs * measure_steps
    sps = total_steps / elapsed
    return {
        "num_envs": num_envs,
        "sps": sps,
        "ms_per_step": 1000.0 * elapsed / measure_steps,
        "device": device,
    }


def run_benchmark_suite(
    device: str = "cpu",
    env_sizes: list[int] | None = None,
    warmup_steps: int = 50,
    measure_steps: int = 500,
) -> None:
    """Run the full benchmark suite and print a report.

    Args:
        device:        Torch device.
        env_sizes:     List of num_envs values to test.
        warmup_steps:  Warmup steps per trial.
        measure_steps: Measured steps per trial.
    """
    if env_sizes is None:
        env_sizes = [1, 16, 64, 256, 512, 1024, 2048, 4096]

    print(f"\n{'='*60}")
    print(f"  HydraLab-USV Throughput Benchmark  [{device.upper()}]")
    print(f"  warmup={warmup_steps} steps  |  measure={measure_steps} steps")
    print(f"{'='*60}")
    print(f"  {'num_envs':>10}  {'SPS':>14}  {'ms/step':>10}")
    print(f"  {'-'*40}")

    results = []
    for n in env_sizes:
        try:
            r = benchmark(n, device, warmup_steps, measure_steps)
            print(f"  {r['num_envs']:>10,}  {r['sps']:>14,.0f}  {r['ms_per_step']:>9.2f}")
            results.append(r)
        except Exception as e:
            print(f"  {n:>10,}  FAILED: {e}")

    if results:
        best = max(results, key=lambda x: x["sps"])
        print(f"\n  Peak: {best['sps']:,.0f} SPS at {best['num_envs']:,} envs")
        print(f"\n  Target (1k envs): ", end="")
        t1k = next((r for r in results if r["num_envs"] >= 1000), None)
        if t1k:
            ok = "✓ PASS" if t1k["sps"] > 0 else "✗ FAIL"
            print(f"{t1k['sps']:,.0f} SPS  {ok}")
        else:
            print("not measured")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="HydraLab-USV Throughput Benchmark")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--max-envs", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--steps", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    import math
    sizes = [2 ** k for k in range(0, int(math.log2(args.max_envs)) + 1)]
    run_benchmark_suite(
        device=args.device,
        env_sizes=sizes,
        warmup_steps=args.warmup,
        measure_steps=args.steps,
    )
