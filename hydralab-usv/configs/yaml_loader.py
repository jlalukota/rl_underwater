"""
YAML configuration loader / saver for HydraLabConfig.

Converts the nested dataclass config tree to/from plain YAML, enabling:
    - experiment configuration files committed to version control
    - CLI overrides of individual fields
    - config versioning and reproducibility

Usage:
    # Save
    from configs.yaml_loader import save_config, load_config
    save_config(cfg, "experiments/run_001.yaml")

    # Load
    cfg = load_config("experiments/run_001.yaml")

    # Merge CLI overrides
    cfg = load_config("base.yaml", overrides={"env.num_envs": 2048})
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from configs.blueboat_cfg import (
    HydraLabConfig,
    BlueboatDynamicsConfig,
    ThrustConfig,
    EnvironmentConfig,
    ObservationConfig,
    DisturbanceConfig,
    DomainRandomizationConfig,
    RewardConfig,
)

# Map sub-config field names to their dataclass types
_SUB_CONFIG_MAP: dict[str, type] = {
    "dynamics": BlueboatDynamicsConfig,
    "thrust": ThrustConfig,
    "env": EnvironmentConfig,
    "obs": ObservationConfig,
    "disturbance": DisturbanceConfig,
    "randomization": DomainRandomizationConfig,
    "reward": RewardConfig,
}


def config_to_dict(cfg: HydraLabConfig) -> dict[str, Any]:
    """Recursively convert a HydraLabConfig to a plain dict."""
    return dataclasses.asdict(cfg)


def dict_to_config(d: dict[str, Any]) -> HydraLabConfig:
    """Reconstruct a HydraLabConfig from a plain dict.

    Handles nested sub-configs and tuple fields.
    """
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(HydraLabConfig):
        name = field.name
        if name not in d:
            continue
        val = d[name]
        if name in _SUB_CONFIG_MAP:
            sub_cls = _SUB_CONFIG_MAP[name]
            # Restore tuples (YAML loads them as lists)
            sub_kwargs = {}
            for sf in dataclasses.fields(sub_cls):
                if sf.name not in val:
                    continue
                fval = val[sf.name]
                if isinstance(sf.default, tuple) or (
                    hasattr(sf, "default_factory")
                    and callable(sf.default_factory)
                    and isinstance(sf.default_factory(), tuple)
                ):
                    fval = tuple(fval) if isinstance(fval, list) else fval
                # Heuristic: if the annotation contains 'Tuple', convert list→tuple
                annotation = sf.type if isinstance(sf.type, str) else str(sf.type)
                if "Tuple" in annotation and isinstance(fval, list):
                    fval = tuple(fval)
                sub_kwargs[sf.name] = fval
            kwargs[name] = sub_cls(**sub_kwargs)
        else:
            kwargs[name] = val
    return HydraLabConfig(**kwargs)


def save_config(cfg: HydraLabConfig, path: str | Path) -> None:
    """Serialise config to YAML file.

    Args:
        cfg:  HydraLabConfig instance.
        path: Destination file path (.yaml).
    """
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required: pip install pyyaml")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_to_dict(cfg), f, default_flow_style=False, sort_keys=True)


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> HydraLabConfig:
    """Load a HydraLabConfig from a YAML file with optional overrides.

    Args:
        path:      Path to YAML config file.
        overrides: Dict of dotted-path overrides, e.g.
                   {"env.num_envs": 2048, "reward.tracking_weight": 3.0}

    Returns:
        Populated HydraLabConfig.
    """
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        raw = yaml.safe_load(f)

    if overrides:
        raw = _apply_overrides(raw, overrides)

    return dict_to_config(raw)


def _apply_overrides(d: dict, overrides: dict[str, Any]) -> dict:
    """Apply dotted-path overrides to a nested dict in-place."""
    import copy
    d = copy.deepcopy(d)
    for dotted_key, val in overrides.items():
        keys = dotted_key.split(".")
        target = d
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = val
    return d


def make_default_config_yaml(path: str | Path = "configs/default.yaml") -> None:
    """Write a default HydraLabConfig YAML to disk for reference."""
    save_config(HydraLabConfig(), path)
    print(f"Default config written to {path}")
