"""
src/utils/config.py
-------------------
Centralised YAML config loader. All other modules import from here
instead of reading YAML files directly.

Usage:
    from src.utils.config import get_config
    cfg = get_config()
    train_path = cfg["paths"]["train_file"]
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Default config location (relative to project root)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> dict[str, Any]:
    """Load and cache the YAML config file.

    Parameters
    ----------
    config_path:
        Override path to a YAML file. Defaults to ``config/config.yaml``
        relative to the project root.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Ensure you are running from the project root directory."
        )

    with path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    return cfg


def get_path(key: str) -> Path:
    """Convenience wrapper: return a config path as a resolved ``Path``.

    Parameters
    ----------
    key:
        Key inside ``cfg["paths"]`` (e.g. ``"train_file"``).

    Returns
    -------
    Path
        Resolved absolute path.
    """
    cfg = get_config()
    raw = cfg["paths"].get(key)
    if raw is None:
        raise KeyError(f"Path key '{key}' not found in config['paths']")
    return Path(raw).resolve()


def reload_config(config_path: str | None = None) -> dict[str, Any]:
    """Force reload the config (clears LRU cache first)."""
    get_config.cache_clear()
    return get_config(config_path)
