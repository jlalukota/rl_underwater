# HydraLab-USV

A research-grade, GPU-accelerated reinforcement learning environment for differential-thrust Autonomous Surface Vessels (ASVs). Built on PyTorch with optional NVIDIA Isaac Lab integration, it simulates a BlueBoat-style catamaran using a 3-DOF Fossen maneuvering model and supports 1000+ parallel environments for high-throughput PPO training.

---

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Physics Model](#physics-model)
- [Observation & Action Space](#observation--action-space)
- [Reward Function](#reward-function)
- [Disturbances & Domain Randomization](#disturbances--domain-randomization)
- [Tasks](#tasks)
- [Training Infrastructure](#training-infrastructure)
- [Setup](#setup)
- [Usage](#usage)
  - [Standalone (Gymnasium)](#standalone-gymnasium)
  - [PPO Training with SB3](#ppo-training-with-sb3)
  - [Curriculum Training](#curriculum-training)
  - [RLlib](#rllib)
  - [Evaluation](#evaluation)
  - [Benchmarking](#benchmarking)
  - [Stability Validation](#stability-validation)
  - [Visualization](#visualization)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Isaac Lab Integration](#isaac-lab-integration)

---

## Overview

HydraLab-USV targets the trajectory-tracking problem for surface vessels operating under realistic marine disturbances (wind, current, stochastic waves). The key design goals are:

- **High throughput**: all dynamics computed in batched PyTorch tensor operations — no Python loops over environments. Achieves ~286k simulated steps/second at 1024 envs on CPU; GPU scales further.
- **Correctness**: implements the full Fossen (2011) maneuvering model with relative-velocity current formulation, RK4 integration, and correct damping sign convention.
- **Curriculum learning**: disturbance scale ramps from near-zero to full difficulty over training to prevent reactive-thrashing policies.
- **SB3 + RLlib ready**: ships wrappers for both Stable-Baselines3 and RLlib with correct terminal-observation handling for value bootstrap at episode boundaries.
- **Dual-mode**: runs standalone (pure gymnasium) for development/CI and under Isaac Lab for photorealistic rendering when Isaac Sim is available.

---

## Architecture

```
HydraLabConfig (dataclass tree)
        │
        ▼
BlueboatEnv  (envs/blueboat_env.py)
    ├── RigidBody3DOF           dynamics/rigid_body.py     RK4 integrator
    │       └── Hydrodynamics   dynamics/hydrodynamics.py  Fossen model
    ├── DifferentialThrustModel dynamics/thrust_model.py   actuator lag
    ├── WindDisturbance         disturbances/wind.py       Gauss-Markov
    ├── CurrentDisturbance      disturbances/current.py    relative-velocity
    ├── WaveDisturbance         disturbances/waves.py      Ornstein-Uhlenbeck
    ├── TrajectoryTrackingTask  tasks/trajectory_tracking.py
    ├── DomainRandomizer        randomization/domain_randomization.py
    ├── EpisodeStats            envs/episode_stats.py
    └── Reward modules          rewards/{tracking,heading,smoothness,energy,boundary}.py
            │
            ▼
HydraLabVecEnvWrapper  (training/ppo_train.py)  ← SB3 VecEnv bridge
            │
            ▼
    Stable-Baselines3 PPO
    + HydraLabMetricsCallback
    + CurriculumCallback
    + BestModelCallback
```

Each environment slot is a fully independent column of tensors of shape `(num_envs, ...)`. There is no Python-level per-environment loop.

---

## Project Structure

```
hydralab-usv/
│
├── configs/
│   ├── blueboat_cfg.py        Master config tree (HydraLabConfig)
│   ├── training_cfg.py        Training + curriculum hyperparameter configs
│   └── yaml_loader.py         YAML ↔ dataclass serialisation helpers
│
├── dynamics/
│   ├── hydrodynamics.py       Fossen 3-DOF model: M·ν̇ + C·ν + D·ν = τ
│   ├── rigid_body.py          RK4 / Euler integrators; state layout
│   └── thrust_model.py        Differential thrust → surge/yaw wrench
│
├── disturbances/
│   ├── wind.py                Gauss-Markov wind (periodic random-walk)
│   ├── current.py             Ocean current → body-frame relative velocity
│   └── waves.py               Ornstein-Uhlenbeck stochastic wave impulses
│
├── rewards/
│   ├── tracking.py            Gaussian position-tracking reward
│   ├── heading.py             Gaussian heading-alignment reward
│   ├── smoothness.py          Action-delta smoothness penalty
│   ├── energy.py              Thrust energy penalty
│   └── boundary.py            Soft + hard workspace boundary penalty
│
├── tasks/
│   ├── trajectory_tracking.py Sinusoidal / lemniscate / circle / spline / straight
│   ├── waypoint_navigation.py Sequential waypoint reaching task
│   └── station_keeping.py     Hold a fixed position against disturbances
│
├── randomization/
│   └── domain_randomization.py  Mass, drag, thruster efficiency, initial state DR
│
├── envs/
│   ├── blueboat_env.py        Main env (dual-mode: standalone or Isaac Lab)
│   ├── episode_stats.py       Per-episode metric accumulator
│   └── isaac_scene.py         Isaac Sim USD scene setup helpers
│
├── training/
│   ├── ppo_train.py           SB3 PPO entry point + HydraLabVecEnvWrapper
│   ├── curriculum.py          DisturbanceCurriculum scheduler
│   ├── callbacks.py           HydraLabMetricsCallback / CurriculumCallback / BestModelCallback
│   ├── rllib_env.py           RLlib single-slot + multi-slot wrappers
│   ├── stability_validator.py Long-run numerical stability checker
│   ├── benchmark.py           Steps-per-second throughput benchmark
│   ├── evaluate.py            Policy evaluation against saved checkpoint
│   └── hyperparams/
│       ├── ppo_default.yaml   Baseline PPO config (no curriculum)
│       └── ppo_curriculum.yaml  Cosine-ramp curriculum config
│
├── visualization/
│   └── debug_renderer.py      Matplotlib 2D top-down debug renderer
│
├── tests/
│   ├── test_dynamics.py       26 unit tests — Fossen model, RK4, thrust
│   ├── test_environment.py    22 integration tests — env reset/step/shapes
│   ├── test_phase2.py         28 tests — disturbances, boundary, episode stats, spline
│   ├── test_phase3.py         30 tests — curriculum, RLlib, stability validator, terminal obs
│   └── test_vectorized.py     12 tests — isolation, auto-reset, action clamping, DR diversity
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Physics Model

### 3-DOF Fossen Maneuvering Model

The rigid-body dynamics follow Fossen (2011):

```
M · ν̇  +  C(ν) · ν  +  D(ν) · ν  =  τ + τ_env
```

| Symbol | Meaning |
|--------|---------|
| `ν = [u, v, r]ᵀ` | surge velocity, sway velocity, yaw rate |
| `M` | system inertia matrix (rigid body + added mass) |
| `C(ν)` | Coriolis-centripetal matrix |
| `D(ν)` | linear + quadratic damping |
| `τ` | thruster wrench `[X, Y, N]ᵀ` |
| `τ_env` | environmental forces (wind + waves) |

**State vector** `(num_envs, 6)`:

| Index | Symbol | Units | Description |
|-------|--------|-------|-------------|
| 0 | x | m | world North position |
| 1 | y | m | world East position |
| 2 | ψ | rad | heading (0 = North, CW positive) |
| 3 | u | m/s | surge (body +X) |
| 4 | v | m/s | sway (body +Y, positive starboard) |
| 5 | r | rad/s | yaw rate |

### BlueBoat Parameters (defaults)

| Parameter | Value | Note |
|-----------|-------|------|
| Mass | 15 kg | hull + payload |
| Yaw inertia | 3.5 kg·m² | |
| Thruster beam | 0.57 m | port–stbd separation |
| Max thrust fwd | 30 N | ~3.06 kgf at 12 V (T200) |
| Max thrust rev | 22 N | asymmetric on T200 |
| Actuator time constant | 0.1 s | first-order lag |
| Linear surge damping Xu | −3.0 N·s/m | |
| Quadratic surge damping Xuu | −5.0 N·s²/m² | |

### Current formulation

Ocean current enters hydrodynamic damping via the **relative velocity** `ν_rel = ν − ν_current`, not as an external force. This is the physically correct formulation: only water-relative motion drives viscous drag.

```python
nu_current = env.current.get_body_frame_velocity(state)  # R(ψ)ᵀ · V_c_world
nu_rel = nu - nu_current
f_damp = hydro.damping_forces(nu_rel)                     # acts on relative motion
```

### Integration

Default: **Runge-Kutta 4** (RK4). Euler is available via `use_rk4=False` on `RigidBody3DOF`. Policy operates at 12.5 Hz (4 physics substeps × 50 Hz physics).

---

## Observation & Action Space

### Observations `(obs_dim = 10)`

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0–1 | x, y position | ÷ 50 m |
| 2–3 | sin(ψ), cos(ψ) | — (unit circle encoding, no ±π discontinuity) |
| 4 | surge u | ÷ 3 m/s |
| 5 | sway v | ÷ 3 m/s |
| 6 | yaw rate r | ÷ π rad/s |
| 7–8 | Δx, Δy to goal | ÷ 50 m |
| 9 | distance to goal | ÷ 50 m |

Optional Gaussian sensor noise is applied at each step (configurable via `ObservationConfig`).

### Actions `(2,)`

```
[a_left, a_right]  ∈ [-1, 1]
```

Normalized thruster commands. Internally mapped to `[−max_thrust_rev, +max_thrust_fwd]` with a first-order actuator lag and dead-band. Symmetric thrust produces pure surge; asymmetric thrust produces yaw moment:

```
X = F_L + F_R               # surge force
N = (F_R − F_L) × beam/2   # yaw moment
```

---

## Reward Function

Composite per-step reward:

```
R = w_track · r_track(pos, goal)
  + w_head  · r_head(ψ, ψ_goal)
  − w_smooth · p_smooth(Δa)
  − w_energy · p_energy(a)
  − w_bound  · p_bound(pos)
```

| Component | Default weight | Shape |
|-----------|---------------|-------|
| Tracking | 2.0 | Gaussian: `exp(−d² / 2σ²)`, σ = 1.0 m |
| Heading | 0.5 | Gaussian on heading error, σ = 0.5 rad |
| Smoothness | 0.1 | Penalty on `‖Δa‖` clamped at 2.0 |
| Energy | 0.05 | Penalty on `‖a‖²` |
| Boundary | 5.0 | Linear ramp beyond 45 m + hard penalty at 50 m |

All weights and shaping parameters are configurable in `RewardConfig`.

---

## Disturbances & Domain Randomization

### Wind (Gauss-Markov)

Periodic random-walk on speed and direction, updated every `wind_update_period` seconds. Produces a drag force proportional to `ρ_air · C_D · A · |V_wind|²`.

Default: `wind_speed_mean = 2.0 m/s`, `wind_speed_std = 0.5 m/s`.

### Ocean Current (relative velocity)

Steady-state current with slow random walk on speed and direction. The current enters dynamics via `ν_rel = ν − ν_current`, so damping forces act on water-relative motion. Direction is randomized per episode.

Default: `current_speed_mean = 0.3 m/s`.

### Waves (Ornstein-Uhlenbeck)

First-order OU process generates correlated stochastic impulses on surge/sway and yaw, representing unmodeled wave forcing:

```
F(t+dt) = decay · F(t) + σ · √(1 − decay²) · N(0, I)
```

Default: `wave_force_std = 1.0 N`, `wave_bandwidth = 0.5 Hz`.

### Domain Randomization (per episode reset)

| Parameter | Default range |
|-----------|--------------|
| Initial position | ±5 m (uniform) |
| Initial heading | ±π rad (uniform) |
| Wind speed | 0–5 m/s |
| Wind direction | 0–2π rad |
| Current speed | 0–1 m/s |
| Vehicle mass | ±10% |
| Drag coefficients | ±20% |
| Thruster efficiency | 80–100% |
| Yaw inertia | ±10% |

---

## Tasks

| Task | Description |
|------|-------------|
| `SINUSOIDAL` | Oscillating reference path; default for training |
| `LEMNISCATE` | Bernoulli figure-8; stresses heading changes |
| `CIRCLE` | Constant curvature; baseline for turning maneuvers |
| `STRAIGHT` | Constant-heading path; simplest baseline |
| `SPLINE` | Catmull-Rom spline through 8 random control points; most general |

All trajectories move at a configurable speed and produce both a target position and a target heading (tangent direction) at each timestep. Switching trajectory type does not require rebuilding the environment.

---

## Training Infrastructure

### Curriculum Learning (`training/curriculum.py`)

Prevents the policy from learning reactive thrashing by starting with near-calm conditions and ramping to full disturbances:

```
Phase A (0 → warmup_steps):    scale = initial_disturbance_scale  (≈ 0.05)
Phase B (warmup → full_steps):  scale ramps via linear / cosine / step schedule
Phase C (full_steps → ∞):       scale = 1.0  (full storm)
```

Domain randomization ramps on a separate (slower) schedule.

### SB3 Callbacks (`training/callbacks.py`)

| Callback | Function |
|----------|----------|
| `HydraLabMetricsCallback` | Logs dist-to-goal, surge, wind speed, current speed, action saturation, boundary violations to TensorBoard |
| `CurriculumCallback` | Applies curriculum schedule to live env every N steps |
| `BestModelCallback` | Saves checkpoint whenever mean episode return improves |

### Terminal Observation Handling

SB3 PPO requires correct value bootstrap at episode boundaries. `BlueboatEnv.step()` captures the **pre-reset terminal observation** and stores it in `info["_terminal_obs"]` before resetting done environments. `HydraLabVecEnvWrapper` moves this into `info[i]["terminal_observation"]` per the SB3 VecEnv contract.

### Stability Validator (`training/stability_validator.py`)

Runs the env for N steps under stress (random/max actions, full disturbances) and checks for NaN/Inf in observations, position/velocity explosion, and reward divergence. Returns a `StabilityReport` with pass/fail status and aggregate statistics.

### RLlib Wrappers (`training/rllib_env.py`)

| Class | Description |
|-------|-------------|
| `HydraLabRLlibEnv` | Single-slot wrapper; one env per RLlib worker |
| `HydraLabRLlibVecEnv` | Multi-slot wrapper; K GPU-batched envs per worker for higher throughput |

---

## Setup

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | `python3 --version` to check |
| git | any | |
| OS | Linux, macOS, or Windows via **WSL2** | WSL2 is the supported path on Windows |
| GPU (optional) | NVIDIA + CUDA-enabled PyTorch | CPU works for development and small runs |

### 1. Clone

```bash
git clone https://github.com/jlalukota/rl_underwater.git
cd rl_underwater/hydralab-usv    # package root: the folder containing setup.py
```

> **WSL users:** clone into the Linux filesystem (e.g. `~/projects`), not `/mnt/c/...`.
> Cross-filesystem I/O is much slower, and OneDrive sync can lock or corrupt files inside `.venv/`.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

If `venv` is missing on Ubuntu/WSL: `sudo apt install python3-venv`.

### 3. Install

Pick the extras you need:

| Command | Installs |
|---|---|
| `pip install -e .` | Core env only (torch, numpy, gymnasium) |
| `pip install -e ".[train]"` | + Stable-Baselines3, TensorBoard |
| `pip install -e ".[viz]"` | + matplotlib |
| `pip install -e ".[all]"` | Everything |

Recommended for first-time setup:

```bash
pip install -e ".[all]"
pip install pyyaml          # enables YAML configs and the 2 YAML-dependent tests
```

If the editable install fails, fall back to:

```bash
pip install -r requirements.txt
pip install -e .
```

**GPU:** install a CUDA build of PyTorch *before* the step above, using the selector at
[pytorch.org/get-started](https://pytorch.org/get-started/locally/). Then verify:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

### 4. Verify the install

```bash
pytest tests/ -v
```

Expected: `116 passed, 3 skipped`. Skips come from missing PyYAML (2) and the SB3 smoke test (1).
With PyYAML installed, the YAML skips go away.

### 5. Smoke test

```bash
# Throughput check (~30 s)
python training/benchmark.py --device cpu --max-envs 256

# Short PPO run to confirm training runs end-to-end
python training/ppo_train.py --num-envs 64 --device cpu --total-steps 100000
```

Checkpoints go to `checkpoints/` and TensorBoard logs go to `logs/ppo_blueboat/`:

```bash
tensorboard --logdir logs/
```

### 6. Verify rendering

The debug renderer uses matplotlib. First check whether a GUI backend is available:

```bash
python -c "import matplotlib; print(matplotlib.get_backend())"
```

If it prints `agg`, no window can open (this is common on headless machines and WSL without WSLg).
**Save to a file instead.** This works everywhere:

```python
# render_check.py
import torch
from configs.blueboat_cfg import HydraLabConfig
from envs.blueboat_env import BlueboatEnv
from visualization.debug_renderer import DebugRenderer

cfg = HydraLabConfig()
cfg.env.num_envs = 4
cfg.device = "cpu"

env = BlueboatEnv(cfg)
env.reset(seed=0)
renderer = DebugRenderer(num_envs_to_show=4)

for _ in range(200):
    actions = torch.rand(4, 2) * 2 - 1
    obs, rew, *_ = env.step(actions)
    renderer.update(env, rew)   # records history only

renderer.render(env)            # actually draws the figure
renderer.save("render_check.png")
print("Saved render_check.png")
env.close()
```

```bash
python render_check.py
explorer.exe render_check.png   # WSL: opens in Windows image viewer
```

On Windows 11 with WSLg, `renderer.show()` will open a live window. On Windows 10 WSL, use the save-to-file path.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: configs` | Running from the wrong directory, or package not installed | `cd` to the folder with `setup.py`, then `pip install -e .` |
| Render window never appears | Non-GUI matplotlib backend (`agg`) | Use `renderer.save(...)` as in step 6 |
| Render shows empty axes | `update()` was called but `render()` wasn't | Call `renderer.render(env)` before `show()`/`save()` |
| `evaluate.py` fails without a checkpoint | It requires a trained PPO model | Train first, or use `render_check.py` |
| `torch.cuda.is_available()` is `False` | CPU-only PyTorch wheel | Reinstall PyTorch from the CUDA selector |
| Very slow installs or tests on WSL | Repo is on `/mnt/c` or OneDrive | Re-clone into `~/` |

## Usage

### Standalone (Gymnasium)

```python
import torch
from configs.blueboat_cfg import HydraLabConfig
from envs.blueboat_env import BlueboatEnv
from tasks.trajectory_tracking import TrajectoryType

cfg = HydraLabConfig()
cfg.env.num_envs = 64
cfg.device = "cuda"   # or "cpu"

env = BlueboatEnv(cfg, traj_type=TrajectoryType.SINUSOIDAL)
obs, info = env.reset(seed=42)

for step in range(1000):
    # obs shape: (64, 10)
    actions = torch.rand(64, 2, device="cuda") * 2 - 1   # random policy
    obs, rewards, terminated, truncated, info = env.step(actions)
    # rewards shape: (64,)

env.close()
```

### PPO Training with SB3

**Quick start (CLI):**

```bash
# CPU, 1024 envs, default hyperparameters
python training/ppo_train.py --num-envs 1024 --device cpu --total-steps 10_000_000

# GPU, lemniscate trajectory
python training/ppo_train.py --num-envs 2048 --device cuda --traj-type lemniscate

# Enable curriculum
python training/ppo_train.py --curriculum --num-envs 1024
```

**From a YAML config:**

```bash
python training/ppo_train.py --config training/hyperparams/ppo_curriculum.yaml
```

**Programmatic:**

```python
from configs.blueboat_cfg import HydraLabConfig
from configs.training_cfg import TrainingConfig, CurriculumConfig
from training.ppo_train import train_ppo

env_cfg = HydraLabConfig()
env_cfg.env.num_envs = 1024
env_cfg.device = "cuda"

train_cfg = TrainingConfig(
    total_timesteps=50_000_000,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=4096,
)
train_cfg.curriculum.enabled = True

train_ppo(env_cfg=env_cfg, train_cfg=train_cfg)
```

TensorBoard logs are written to `logs/ppo_blueboat/` by default:

```bash
tensorboard --logdir logs/
```

Key logged metrics under the `marine/` prefix:

| Metric | Description |
|--------|-------------|
| `marine/mean_dist_to_goal_m` | Mean distance to reference point |
| `marine/mean_surge_mps` | Mean surge velocity |
| `marine/wind_speed_mean` | Current wind speed |
| `marine/current_speed_mean` | Current speed |
| `marine/action_saturation_frac` | Fraction of thrusters at ±1.0 |
| `marine/boundary_violation_frac` | Fraction of envs outside workspace |
| `curriculum/disturbance_scale` | Current curriculum scale (0→1) |

### Curriculum Training

The curriculum config controls the disturbance ramp:

```python
from configs.training_cfg import CurriculumConfig

# Cosine ramp: slow start, fast middle, gradual finish
curriculum_cfg = CurriculumConfig(
    enabled=True,
    schedule="cosine",        # "linear", "cosine", or "step"
    warmup_steps=500_000,     # steps at initial (calm) difficulty
    full_steps=5_000_000,     # steps when full difficulty is reached
    initial_disturbance_scale=0.05,   # ~5% of nominal disturbances
    final_disturbance_scale=1.0,
    initial_dr_scale=0.0,     # DR disabled during warmup
    final_dr_scale=1.0,
)
```

Or use the provided YAML directly:

```bash
python training/ppo_train.py --config training/hyperparams/ppo_curriculum.yaml
```

### RLlib

```python
from ray.tune import register_env
from training.rllib_env import HydraLabRLlibEnv

register_env("HydraLabUSV-v0", lambda cfg: HydraLabRLlibEnv(cfg))

from ray.rllib.algorithms.ppo import PPOConfig

algo = (
    PPOConfig()
    .environment("HydraLabUSV-v0", env_config={"device": "cpu"})
    .rollouts(num_rollout_workers=8, num_envs_per_worker=1)
    .build()
)

for _ in range(100):
    result = algo.train()
    print(result["episode_reward_mean"])
```

For higher throughput with GPU-batched workers:

```python
from training.rllib_env import HydraLabRLlibVecEnv

register_env("HydraLabUSV-Vec-v0",
    lambda cfg: HydraLabRLlibVecEnv(cfg))

# Each worker holds 16 env slots internally via batched GPU ops
algo = (
    PPOConfig()
    .environment("HydraLabUSV-Vec-v0", env_config={"slots": 16, "device": "cuda"})
    .rollouts(num_rollout_workers=4)
    .build()
)
```

### Evaluation

```bash
# Evaluate a saved SB3 PPO checkpoint
python training/evaluate.py \
    --checkpoint checkpoints/ppo_blueboat_final \
    --n-episodes 200 \
    --device cuda

# With debug rendering
python training/evaluate.py \
    --checkpoint checkpoints/best_model \
    --render
```

Programmatic:

```python
from training.evaluate import evaluate_policy, evaluate_random_policy
from configs.blueboat_cfg import HydraLabConfig

cfg = HydraLabConfig()
cfg.env.num_envs = 32
cfg.device = "cpu"

# Evaluate trained policy
result = evaluate_policy("checkpoints/ppo_blueboat_final", cfg=cfg, n_episodes=100)
print(f"Mean return: {result['mean_return']:.2f}")
print(f"Mean tracking error: {result['mean_tracking_error_m']:.3f} m")

# Evaluate random baseline
evaluate_random_policy(cfg, n_episodes=50)
```

### Benchmarking

```bash
# CPU throughput across env counts
python training/benchmark.py --device cpu --max-envs 4096

# GPU throughput
python training/benchmark.py --device cuda --max-envs 8192
```

Example output:

```
============================================================
  HydraLab-USV Throughput Benchmark  [CPU]
  warmup=50 steps  |  measure=500 steps
============================================================
    num_envs             SPS     ms/step
  ----------------------------------------
           1           3,421       0.29
          16          31,850       0.50
          64         107,213       0.60
         256         209,441       1.22
         512         253,820       2.02
       1,024         286,147       3.58
       2,048         301,203       6.80

  Peak: 301,203 SPS at 2,048 envs
============================================================
```

### Stability Validation

Validates that a given config produces numerically stable rollouts before committing to a long training run:

```python
from training.stability_validator import StabilityValidator
from configs.blueboat_cfg import HydraLabConfig

cfg = HydraLabConfig()
cfg.env.num_envs = 64
cfg.device = "cpu"

validator = StabilityValidator(
    cfg=cfg,
    n_steps=50_000,     # steps to simulate
    action_mode="random",   # "random", "max", or "zero"
    max_pos_m=500.0,    # fail if vessel exceeds this distance
    max_vel=50.0,       # fail if any velocity component exceeds this
)
report = validator.run()
print(report.summary())
# StabilityReport(PASS)
#   steps run      : 50000
#   max |pos|      : 47.23 m
#   max |vel|      : 2.98 m/s (or rad/s)
#   max |reward|   : 2.01
```

### Visualization

```python
from configs.blueboat_cfg import HydraLabConfig
from envs.blueboat_env import BlueboatEnv
from visualization.debug_renderer import DebugRenderer
import torch

cfg = HydraLabConfig()
cfg.env.num_envs = 4
cfg.device = "cpu"

env = BlueboatEnv(cfg)
env.reset()

renderer = DebugRenderer(num_envs_to_show=4)

for _ in range(500):
    actions = torch.zeros(4, 2)
    obs, rew, *_ = env.step(actions)
    renderer.update(env)

renderer.show()
```

---

## Configuration Reference

All configuration is done via Python dataclasses. Every field has a documented default.

### HydraLabConfig (top-level)

```python
from configs.blueboat_cfg import HydraLabConfig

cfg = HydraLabConfig()
cfg.device = "cuda"       # "cpu" or "cuda"
cfg.seed = 42
```

### Key sub-configs

#### EnvironmentConfig

```python
cfg.env.num_envs = 1024
cfg.env.dt = 0.02              # physics timestep (s) → 50 Hz
cfg.env.decimation = 4         # physics steps per policy step → 12.5 Hz policy
cfg.env.episode_length_s = 60  # seconds per episode
cfg.env.workspace_radius = 50  # m — reset if vessel leaves this radius
cfg.env.max_surge = 3.0        # m/s — velocity clamped to this
```

#### DisturbanceConfig

```python
cfg.disturbance.wind_speed_mean = 2.0      # m/s
cfg.disturbance.current_speed_mean = 0.3   # m/s
cfg.disturbance.wave_force_std = 1.0       # N
cfg.disturbance.wind_enabled = True
cfg.disturbance.current_enabled = True
cfg.disturbance.wave_enabled = True
```

#### RewardConfig

```python
cfg.reward.tracking_weight = 2.0
cfg.reward.tracking_sigma = 1.0     # Gaussian bandwidth in meters
cfg.reward.heading_weight = 0.5
cfg.reward.boundary_weight = 5.0
cfg.reward.boundary_margin = 5.0    # soft penalty begins 5 m inside workspace edge
```

#### DomainRandomizationConfig

```python
cfg.randomization.enabled = True
cfg.randomization.mass_noise_frac = 0.10       # ±10% mass
cfg.randomization.drag_noise_frac = 0.20       # ±20% drag
cfg.randomization.thrust_efficiency_range = (0.8, 1.0)
```

### TrainingConfig

```python
from configs.training_cfg import TrainingConfig, CurriculumConfig

train_cfg = TrainingConfig(
    total_timesteps=50_000_000,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=4096,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    log_dir="logs/my_run",
    checkpoint_dir="checkpoints/my_run",
)
train_cfg.curriculum = CurriculumConfig(
    enabled=True,
    schedule="cosine",
    warmup_steps=500_000,
    full_steps=5_000_000,
)
```

### YAML config loading

```python
from training.ppo_train import load_training_config

train_cfg = load_training_config("training/hyperparams/ppo_curriculum.yaml")
```

Or save an existing config:

```python
from configs.yaml_loader import save_config
save_config(cfg, "my_env_config.yaml")
```

---

## Testing

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/test_dynamics.py -v      # 26 physics unit tests
pytest tests/test_environment.py -v   # 22 env integration tests
pytest tests/test_phase2.py -v        # 28 disturbance/reward/task tests
pytest tests/test_phase3.py -v        # 30 curriculum/RLlib/stability tests
pytest tests/test_vectorized.py -v    # 12 isolation/auto-reset tests

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

Expected result: **116 passed, 3 skipped** (2 skipped require PyYAML, 1 requires SB3).

### Notable test groups

| Group | What it covers |
|-------|---------------|
| `TestDeceleration` | Vessel slows under drag with no thrust (critical Fossen sign check) |
| `TestCoriolis` | Sway+yaw coupling produces correct surge force |
| `TestIsolation` | State changes in env 0 do not affect env 1 |
| `TestAutoReset` | Terminal obs is pre-reset; returned obs is post-reset |
| `TestCurriculumScale` | Linear/cosine/step schedules reach correct values |
| `TestStabilityValidator` | Detects position explosion, passes on calm env |
| `TestRLlibEnv` | Step returns numpy arrays with correct dtype and shape |

---

## Isaac Lab Integration

When `omni.isaac.lab` is importable, `BlueboatEnv` inherits from `DirectRLEnv` instead of the gymnasium fallback:

```
Isaac Lab mode: BlueboatEnv → DirectRLEnv
    _setup_scene()       — registers USD vessel asset + water surface
    _pre_physics_step()  — caches clamped actions
    _apply_action()      — integrates marine dynamics (bypasses PhysX)
    _get_observations()  — returns normalised obs tensor
    _get_rewards()       — returns composite reward tensor
    _get_dones()         — boundary + NaN termination
    _reset_idx()         — per-env stochastic reset
```

Marine dynamics bypass PhysX entirely — all force computation happens in our PyTorch layer. Isaac Sim is used for rendering only.

**To enable Isaac Lab mode**, install Isaac Sim following the [official guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/pip_installation.html), then run normally. No code changes are required.

**Current stubs** (to fill in for full Isaac Sim deployment):
- `envs/isaac_scene.py` — USD asset path for the BlueBoat hull mesh
- Goal marker visual prim registration in `_setup_scene()`

---

## References

- Fossen, T. I. (2011). *Handbook of Marine Craft Hydrodynamics and Motion Control*. Wiley.
- BlueRobotics BlueBoat specifications and T200 thruster thrust curves.
- Barry, P. J. & Goldman, R. N. (1988). *A recursive evaluation algorithm for a class of Catmull-Rom splines.* ACM SIGGRAPH.
- NVIDIA Isaac Lab documentation: https://isaac-sim.github.io/IsaacLab/
