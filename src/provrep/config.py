"""Configuration loading with single-level inheritance and deep merging."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict[str, Any]:
    """Load a YAML config. A top-level ``inherit: other.yaml`` key is resolved
    relative to the file and merged underneath it."""
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    """Set ``a.b.c=value`` in-place (used by CLI ``--set`` overrides)."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(value) if isinstance(value, str) else value
