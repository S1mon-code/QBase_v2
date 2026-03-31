"""Optimizer constants and configuration."""

from __future__ import annotations

# Trade count minimums by frequency
MIN_TRADES: dict[str, int] = {
    "daily": 10,
    "4h": 20,
    "2h": 25,
    "1h": 30,
    "30min": 50,
    "15min": 80,
    "10min": 80,
    "5min": 80,
}

# Objective weights (sum = 1.0)
WEIGHTS: dict[str, float] = {
    "performance": 0.40,
    "significance": 0.15,
    "consistency": 0.15,
    "risk": 0.15,
    "alpha": 0.15,
}

# Known parameter ranges (exact bounds from domain knowledge)
KNOWN_RANGES: dict[str, tuple[float, float]] = {
    "atr_trail_mult": (2.0, 5.5),
    "atr_stop_mult": (2.0, 5.5),
    "st_mult": (1.5, 5.0),
    "kc_mult": (1.0, 3.0),
    "chandelier_mult": (1.5, 4.0),
    "psar_af_step": (0.01, 0.04),
    "psar_af_max": (0.1, 0.3),
}

# Parameters to skip during discovery
SKIP_PARAMS: frozenset[str] = frozenset({
    "name", "warmup", "regime", "horizon", "signal_dimensions", "freq",
})

# Phase-specific trial counts
COARSE_TRIALS: int = 30
FINE_TRIALS: int = 50
PROBE_TRIALS: int = 5

# Robustness settings
NARROW_RADIUS: float = 0.15
ROBUSTNESS_THRESHOLD: float = 0.60  # >=60% neighbors above threshold

# Multi-seed defaults
DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 456)
