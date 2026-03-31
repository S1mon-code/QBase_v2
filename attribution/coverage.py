"""Regime Coverage Matrix + RED FLAG detection.

Checks whether a portfolio of strategies covers all market regimes.
Raises RED FLAGS when all strategies lose money in any single regime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageResult:
    """Result of regime coverage analysis."""

    matrix: dict[str, dict[str, float]]  # {strategy: {regime: pnl}}
    red_flags: tuple[str, ...]  # regimes where ALL strategies have negative PnL
    coverage_score: float  # fraction of regimes with at least one positive strategy


def regime_coverage(
    strategy_regime_pnls: dict[str, dict[str, float]],
) -> CoverageResult:
    """Check if portfolio covers all regimes. RED FLAG if any regime all negative.

    Args:
        strategy_regime_pnls: Nested dict mapping strategy name to a dict of
                             {regime: total_pnl}. E.g.:
                             {"strat_a": {"trend": 100, "mr": -50},
                              "strat_b": {"trend": -20, "mr": 80}}

    Returns:
        CoverageResult with coverage matrix, red flags, and coverage score.
    """
    if not strategy_regime_pnls:
        return CoverageResult(
            matrix={},
            red_flags=(),
            coverage_score=0.0,
        )

    # Collect all regime labels
    all_regimes: set[str] = set()
    for regime_pnls in strategy_regime_pnls.values():
        all_regimes.update(regime_pnls.keys())

    sorted_regimes = sorted(all_regimes)

    # Find red flags and compute coverage
    red_flags: list[str] = []
    covered_count = 0

    for regime in sorted_regimes:
        has_positive = False
        all_negative = True
        for regime_pnls in strategy_regime_pnls.values():
            pnl = regime_pnls.get(regime, 0.0)
            if pnl > 0.0:
                has_positive = True
                all_negative = False
                break
            if pnl >= 0.0:
                all_negative = False

        if has_positive:
            covered_count += 1
        if all_negative:
            red_flags.append(regime)

    total_regimes = len(sorted_regimes)
    coverage_score = covered_count / total_regimes if total_regimes > 0 else 0.0

    return CoverageResult(
        matrix=dict(strategy_regime_pnls),
        red_flags=tuple(red_flags),
        coverage_score=coverage_score,
    )
