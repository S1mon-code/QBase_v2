"""Monitoring Dashboard — Status Summary.

Generates a structured overview of strategy and portfolio health
for display or logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from monitoring.decay_detector import run_all_checks
from monitoring.retirement import monitor_strategy_health


@dataclass(frozen=True)
class StrategyStatus:
    """Health status of a single strategy.

    Attributes
    ----------
    name : str
        Strategy identifier.
    regime : str
        Active regime label.
    direction : str
        Current directional bias (``"long"``, ``"short"``, ``"neutral"``).
    current_sharpe_60d : float | None
        60-day rolling Sharpe ratio, ``None`` if insufficient data.
    alerts : tuple[str, ...]
        Alert messages from decay detection.
    retirement_status : str
        ``"normal"``, ``"observe"``, ``"remove"``, or ``"immediate_remove"``.
    """

    name: str
    regime: str
    direction: str
    current_sharpe_60d: float | None
    alerts: tuple[str, ...]
    retirement_status: str


@dataclass(frozen=True)
class DashboardSummary:
    """Portfolio-level monitoring dashboard.

    Attributes
    ----------
    instrument : str
        Instrument symbol.
    active_regime : str
        Currently assigned regime.
    n_active_strategies : int
        Number of active strategies.
    strategies : tuple[StrategyStatus, ...]
        Per-strategy status entries.
    portfolio_dd : float | None
        Current portfolio drawdown (negative).
    stop_level : str
        ``"normal"``, ``"warning"``, ``"reduce"``, or ``"circuit"``.
    alerts : tuple[str, ...]
        Portfolio-level alert messages.
    """

    instrument: str
    active_regime: str
    n_active_strategies: int
    strategies: tuple[StrategyStatus, ...]
    portfolio_dd: float | None
    stop_level: str
    alerts: tuple[str, ...]


def _compute_sharpe_60d(daily_returns: np.ndarray) -> float | None:
    """Compute 60-day rolling Sharpe from the tail of daily returns.

    Returns
    -------
    float | None
        Annualised Sharpe or ``None`` if insufficient data.
    """
    if len(daily_returns) < 60:
        return None
    window = daily_returns[-60:]
    mean = float(np.mean(window))
    std = float(np.std(window, ddof=1))
    if std == 0.0:
        return 0.0
    return (mean / std) * np.sqrt(252)


def generate_dashboard(
    instrument: str,
    strategy_returns: dict[str, np.ndarray] | None = None,
    regime: str = "unknown",
    direction: str = "neutral",
    portfolio_dd: float | None = None,
    stop_level: str = "normal",
) -> DashboardSummary:
    """Generate monitoring dashboard summary.

    Returns a basic structure even when live data is unavailable.

    Parameters
    ----------
    instrument : str
        Instrument symbol (e.g. ``"RB"``).
    strategy_returns : dict[str, np.ndarray] | None
        Mapping of strategy name to daily returns array.
        ``None`` produces an empty dashboard.
    regime : str
        Active regime label.
    direction : str
        Current directional bias.
    portfolio_dd : float | None
        Current portfolio drawdown.
    stop_level : str
        Current portfolio stop level.

    Returns
    -------
    DashboardSummary
        Structured dashboard output.
    """
    if strategy_returns is None:
        return DashboardSummary(
            instrument=instrument,
            active_regime=regime,
            n_active_strategies=0,
            strategies=(),
            portfolio_dd=portfolio_dd,
            stop_level=stop_level,
            alerts=(),
        )

    strategy_statuses: list[StrategyStatus] = []
    all_alerts: list[str] = []

    for name, daily_rets in strategy_returns.items():
        daily_rets = np.asarray(daily_rets, dtype=float)

        # Decay checks
        decay_alerts = run_all_checks(daily_returns=daily_rets)
        alert_msgs = tuple(a.message for a in decay_alerts)

        # 60d Sharpe
        sharpe_60d = _compute_sharpe_60d(daily_rets)

        # Collect portfolio-level alerts
        for msg in alert_msgs:
            all_alerts.append(f"[{name}] {msg}")

        strategy_statuses.append(
            StrategyStatus(
                name=name,
                regime=regime,
                direction=direction,
                current_sharpe_60d=sharpe_60d,
                alerts=alert_msgs,
                retirement_status="normal",
            )
        )

    return DashboardSummary(
        instrument=instrument,
        active_regime=regime,
        n_active_strategies=len(strategy_statuses),
        strategies=tuple(strategy_statuses),
        portfolio_dd=portfolio_dd,
        stop_level=stop_level,
        alerts=tuple(all_alerts),
    )
