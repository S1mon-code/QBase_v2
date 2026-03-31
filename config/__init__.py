"""Configuration loader for QBase_v2."""

from pathlib import Path
from typing import Any

import yaml


_CONFIG_DIR = Path(__file__).resolve().parent
_cache: dict[str, Any] = {}


def _load_yaml(name: str) -> dict:
    """Load and cache a YAML config file."""
    if name not in _cache:
        path = _CONFIG_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            _cache[name] = yaml.safe_load(f)
    return _cache[name]


def get_settings() -> dict:
    """Load global settings."""
    return _load_yaml("settings.yaml")


def get_fundamental_views() -> dict:
    """Load current fundamental directional views."""
    return _load_yaml("fundamental_views.yaml")


def get_regime_thresholds(instrument: str | None = None) -> dict:
    """Load regime labeling thresholds, with optional per-instrument overrides."""
    raw = _load_yaml("regime_thresholds.yaml")
    base = dict(raw["default"])

    if instrument and "overrides" in raw and raw["overrides"]:
        overrides = raw["overrides"].get(instrument, {})
        if overrides:
            base.update(overrides)

    return base


def get_alphaforge_path() -> Path:
    """Return AlphaForge root path."""
    return Path(get_settings()["alphaforge_path"])


def get_data_dir() -> Path:
    """Return AlphaForge data directory."""
    return Path(get_settings()["data_dir"])


def get_instruments() -> list[str]:
    """Return list of target instruments."""
    return get_settings()["instruments"]


def get_frequencies() -> list[str]:
    """Return list of target frequencies."""
    return get_settings()["frequencies"]


def clear_cache() -> None:
    """Clear config cache (for testing)."""
    _cache.clear()
