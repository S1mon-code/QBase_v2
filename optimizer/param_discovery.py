"""Auto-discover optimizable parameters from strategy class annotations."""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

from optimizer.config import KNOWN_RANGES, SKIP_PARAMS


def _is_period_param(name: str) -> bool:
    """Check if parameter name suggests a period/lookback."""
    return "period" in name or "lookback" in name


def _is_threshold_or_mult(name: str) -> bool:
    """Check if parameter name suggests a threshold or multiplier."""
    return "mult" in name or "threshold" in name or "factor" in name


def _infer_range(
    name: str, param_type: type, default: Any,
) -> dict[str, Any]:
    """Infer parameter range from name, type, and default value.

    Rules:
    - Known params → fixed ranges from KNOWN_RANGES.
    - Period/lookback params → default * [0.4, 3.0].
    - Threshold/mult params → default * [0.3, 3.0].
    - Other int → default * [0.4, 3.0], step=1.
    - Other float → default * [0.3, 3.0], step=None (continuous).
    """
    # Known parameter with exact range
    if name in KNOWN_RANGES:
        low, high = KNOWN_RANGES[name]
        step = None if param_type is float else 1
        return {"type": param_type, "default": default, "low": low, "high": high, "step": step}

    # Guard: default must be numeric and non-zero for range inference
    if not isinstance(default, (int, float)) or default == 0:
        low = 0 if param_type is int else 0.0
        high = 10 if param_type is int else 10.0
        step = 1 if param_type is int else None
        return {"type": param_type, "default": default, "low": low, "high": high, "step": step}

    if _is_period_param(name):
        low = param_type(default * 0.4)
        high = param_type(default * 3.0)
        step = 1 if param_type is int else None
        return {"type": param_type, "default": default, "low": low, "high": high, "step": step}

    if _is_threshold_or_mult(name):
        low = param_type(default * 0.3)
        high = param_type(default * 3.0)
        step = 1 if param_type is int else None
        return {"type": param_type, "default": default, "low": low, "high": high, "step": step}

    # Generic numeric parameter
    if param_type is int:
        low = max(1, int(default * 0.4))
        high = int(default * 3.0)
        return {"type": int, "default": default, "low": low, "high": high, "step": 1}

    if param_type is float:
        low = default * 0.3
        high = default * 3.0
        return {"type": float, "default": default, "low": low, "high": high, "step": None}

    # Fallback for unknown types
    return {"type": param_type, "default": default, "low": 0, "high": 10, "step": None}


def discover_params(strategy_class: type) -> dict[str, dict]:
    """Discover optimizable parameters from type annotations + defaults.

    Returns:
        {param_name: {"type": int/float, "default": val, "low": low, "high": high, "step": step}}

    Rules:
    - Skip: name, warmup, regime, horizon, signal_dimensions, freq
    - Known params: get fixed ranges from KNOWN_RANGES
    - Period params (contains 'period'/'lookback'): default * [0.4, 3.0]
    - Threshold/mult params: default * [0.3, 3.0]
    - int step = 1, float step = None (continuous)
    """
    result: dict[str, dict] = {}

    # Get type hints (handles forward refs)
    try:
        hints = get_type_hints(strategy_class)
    except Exception:
        hints = getattr(strategy_class, "__annotations__", {})

    for param_name, param_type in hints.items():
        # Skip non-optimizable parameters
        if param_name in SKIP_PARAMS:
            continue

        # Skip private/dunder attributes
        if param_name.startswith("_"):
            continue

        # Only optimize int and float parameters
        if param_type not in (int, float):
            continue

        # Get default value
        default = getattr(strategy_class, param_name, None)
        if default is None:
            # Try __init__ defaults
            sig = inspect.signature(strategy_class)
            param = sig.parameters.get(param_name)
            if param is not None and param.default is not inspect.Parameter.empty:
                default = param.default
            else:
                continue

        result[param_name] = _infer_range(param_name, param_type, default)

    return result
