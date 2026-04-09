"""Load validation thresholds from config.yaml."""

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_cache: dict | None = None


def get_thresholds() -> dict:
    """Return cached validation thresholds from config.yaml."""
    global _cache
    if _cache is None:
        with open(_CONFIG_PATH) as f:
            _cache = yaml.safe_load(f)
    return _cache
