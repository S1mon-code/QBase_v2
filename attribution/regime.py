"""Layer C: Regime Attribution.

Aggregates trade PnL by regime labels to understand which market regimes
the strategy performs best and worst in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegimeStats:
    """Per-regime trade statistics."""

    regime: str
    n_trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float


@dataclass(frozen=True)
class RegimeAttributionResult:
    """Aggregated regime attribution across all regimes."""

    stats: tuple[RegimeStats, ...]
    best_regime: str
    worst_regime: str
    regime_dependent: bool  # True if best/worst win_rate differ by > 30pp


def regime_attribution(
    trade_pnls: np.ndarray,
    trade_regimes: np.ndarray,
) -> RegimeAttributionResult:
    """Aggregate trade PnL by regime label.

    Args:
        trade_pnls: 1-D array of per-trade PnL values.
        trade_regimes: 1-D string array of regime labels per trade.

    Returns:
        RegimeAttributionResult with per-regime stats and best/worst identification.

    Raises:
        ValueError: If arrays are empty or have mismatched lengths.
    """
    pnls = np.asarray(trade_pnls, dtype=np.float64).ravel()
    regimes = np.asarray(trade_regimes).ravel()

    if len(pnls) == 0:
        raise ValueError("trade_pnls must not be empty")
    if len(pnls) != len(regimes):
        raise ValueError("trade_pnls and trade_regimes must have the same length")

    unique_regimes = sorted(set(regimes.tolist()))
    stats_list: list[RegimeStats] = []

    for regime in unique_regimes:
        mask = regimes == regime
        regime_pnls = pnls[mask]
        n_trades = int(len(regime_pnls))
        wins = int(np.sum(regime_pnls > 0))
        win_rate = wins / n_trades if n_trades > 0 else 0.0
        avg_pnl = float(np.mean(regime_pnls))
        total_pnl = float(np.sum(regime_pnls))

        stats_list.append(
            RegimeStats(
                regime=str(regime),
                n_trades=n_trades,
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                total_pnl=total_pnl,
            )
        )

    stats_tuple = tuple(stats_list)

    best = max(stats_list, key=lambda s: s.win_rate)
    worst = min(stats_list, key=lambda s: s.win_rate)

    regime_dependent = (best.win_rate - worst.win_rate) > 0.30

    return RegimeAttributionResult(
        stats=stats_tuple,
        best_regime=best.regime,
        worst_regime=worst.regime,
        regime_dependent=regime_dependent,
    )
