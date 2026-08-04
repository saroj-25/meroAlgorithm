"""Configuration loading.

A run is reproducible when it is fully described by (code, config, seed).
Every entry-point script therefore takes ``--config configs/xxx.yaml`` and
nothing else that changes behaviour.

Usage
-----
>>> cfg = load_config("configs/default.yaml")
>>> cfg.get("retrieval.rrf_k")
60
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Repository root = two levels above this file (src/utils/config.py -> repo/)
ROOT = Path(__file__).resolve().parents[2]


class Config:
    """Thin dict wrapper that supports dotted lookups and path resolution."""

    def __init__(self, data: Dict[str, Any], source: str | None = None):
        self._data = data
        self.source = source

    # ------------------------------------------------------------------ access
    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    # ------------------------------------------------------------------- paths
    def path(self, dotted_key: str) -> Path:
        """Resolve a configured path relative to the repository root."""
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"No path configured at '{dotted_key}'")
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    def ensure_dirs(self) -> None:
        for key in ("paths.processed_dir", "paths.checkpoints_dir", "paths.results_dir",
                    "paths.figures_dir", "paths.tables_dir", "paths.logs_dir",
                    "paths.raw_dir"):
            if self.get(key):
                self.path(key).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helper
    def copy_with(self, **overrides: Any) -> "Config":
        """Return a copy with dotted-key overrides applied (used by sweeps)."""
        clone = Config(self.as_dict(), self.source)
        for key, value in overrides.items():
            clone.set(key.replace("__", "."), value)
        return clone

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(source={self.source!r}, keys={list(self._data)})"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | os.PathLike = "configs/default.yaml") -> Config:
    """Load a YAML config, resolving a single-level ``inherit:`` chain."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    parent = data.pop("inherit", None)
    if parent:
        base = load_config(parent).as_dict()
        data = _deep_merge(base, data)

    cfg = Config(data, source=str(p))
    cfg.ensure_dirs()
    return cfg
