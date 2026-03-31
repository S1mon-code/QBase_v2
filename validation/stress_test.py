"""Layer 6b: Stress Testing.

Tests strategy robustness under adverse conditions: doubled slippage,
doubled costs, adjacent timeframes, and similar instruments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StressTestResult:
    """Result of stress tests."""

    slippage_sensitivity: str  # "LOW" / "MODERATE" / "HIGH"
    slippage_decay_pct: float
    cost_doubled_sharpe: float | None
    adjacent_freq_sharpe: float | None
    similar_instrument_sharpe: float | None


def check_slippage_sensitivity(
    base_sharpe: float,
    doubled_sharpe: float,
) -> tuple[str, float]:
    """Check sensitivity to doubled slippage.

    Sensitivity levels:
        - "LOW": decay < 15%
        - "MODERATE": 15% <= decay < 30%
        - "HIGH": decay >= 30%

    Args:
        base_sharpe: Sharpe under normal slippage.
        doubled_sharpe: Sharpe under doubled slippage.

    Returns:
        Tuple of (sensitivity_level, decay_pct).
    """
    if base_sharpe == 0.0:
        decay_pct = 0.0 if doubled_sharpe == 0.0 else 100.0
    else:
        decay_pct = (base_sharpe - doubled_sharpe) / abs(base_sharpe) * 100.0

    if decay_pct < 15.0:
        level = "LOW"
    elif decay_pct < 30.0:
        level = "MODERATE"
    else:
        level = "HIGH"

    return level, decay_pct


def run_stress_test(
    base_sharpe: float,
    doubled_slippage_sharpe: float,
    cost_doubled_sharpe: float | None = None,
    adjacent_freq_sharpe: float | None = None,
    similar_instrument_sharpe: float | None = None,
) -> StressTestResult:
    """Run stress tests and aggregate results.

    Args:
        base_sharpe: Sharpe under normal conditions.
        doubled_slippage_sharpe: Sharpe with doubled slippage.
        cost_doubled_sharpe: Sharpe with doubled transaction costs.
        adjacent_freq_sharpe: Sharpe on adjacent timeframe.
        similar_instrument_sharpe: Sharpe on similar instrument.

    Returns:
        StressTestResult with sensitivity levels and optional metrics.
    """
    sensitivity, decay_pct = check_slippage_sensitivity(
        base_sharpe, doubled_slippage_sharpe
    )

    return StressTestResult(
        slippage_sensitivity=sensitivity,
        slippage_decay_pct=decay_pct,
        cost_doubled_sharpe=cost_doubled_sharpe,
        adjacent_freq_sharpe=adjacent_freq_sharpe,
        similar_instrument_sharpe=similar_instrument_sharpe,
    )
