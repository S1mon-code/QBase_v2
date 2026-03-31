"""Portfolio Stop Loss — re-export with portfolio-specific defaults.

Wraps ``risk.portfolio_stops.PortfolioStops`` for use within the portfolio module.
"""

from __future__ import annotations

from risk.portfolio_stops import PortfolioStops


def create_portfolio_stops(
    warning: float = -0.10,
    reduce: float = -0.15,
    circuit: float = -0.20,
    daily: float = -0.05,
) -> PortfolioStops:
    """Create a PortfolioStops instance with portfolio-specific defaults.

    Parameters
    ----------
    warning : float
        Drawdown warning threshold.
    reduce : float
        Drawdown reduction threshold.
    circuit : float
        Drawdown circuit breaker threshold.
    daily : float
        Daily loss circuit breaker threshold.

    Returns
    -------
    PortfolioStops
        Configured stops instance.
    """
    return PortfolioStops(
        warning=warning,
        reduce=reduce,
        circuit=circuit,
        daily=daily,
    )


__all__ = ["PortfolioStops", "create_portfolio_stops"]
