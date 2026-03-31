"""Strategy Retirement Monitor.

Thin wrapper that computes rolling metrics from monthly returns
and delegates to ``portfolio.retirement.check_retirement``.
"""

from __future__ import annotations

import numpy as np

from portfolio.retirement import RetirementCheck, check_retirement


def _rolling_sharpe(monthly_returns: np.ndarray, months: int) -> float:
    """Compute annualised Sharpe from the last *months* of monthly returns.

    Parameters
    ----------
    monthly_returns : np.ndarray
        1-D array of monthly returns.
    months : int
        Number of trailing months to use.

    Returns
    -------
    float
        Annualised Sharpe ratio.  Returns 0.0 when std is zero.
    """
    window = monthly_returns[-months:]
    mean = float(np.mean(window))
    std = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
    if std == 0.0:
        return 0.0
    return (mean / std) * np.sqrt(12)


def _consecutive_loss_months(monthly_returns: np.ndarray) -> int:
    """Count consecutive negative months from the end of the array.

    Parameters
    ----------
    monthly_returns : np.ndarray
        1-D array of monthly returns.

    Returns
    -------
    int
        Number of consecutive loss months at the tail.
    """
    count = 0
    for ret in reversed(monthly_returns):
        if ret < 0:
            count += 1
        else:
            break
    return count


def monitor_strategy_health(
    strategy: str,
    monthly_returns: np.ndarray,
    current_dd: float,
    backtest_max_dd: float,
) -> RetirementCheck:
    """Compute rolling metrics and check retirement criteria.

    Calculates 6-month and 12-month rolling Sharpe ratios and consecutive
    loss months from ``monthly_returns``, then delegates to
    :func:`portfolio.retirement.check_retirement`.

    Parameters
    ----------
    strategy : str
        Strategy name.
    monthly_returns : np.ndarray
        1-D array of at least 12 monthly returns.
    current_dd : float
        Current drawdown (negative, e.g. -0.15).
    backtest_max_dd : float
        Maximum drawdown from backtest (negative, e.g. -0.10).

    Returns
    -------
    RetirementCheck
        Retirement decision from the portfolio module.
    """
    monthly_returns = np.asarray(monthly_returns, dtype=float)

    n_months = len(monthly_returns)
    sharpe_6m = _rolling_sharpe(monthly_returns, min(6, n_months))
    sharpe_12m = _rolling_sharpe(monthly_returns, min(12, n_months))
    consec_losses = _consecutive_loss_months(monthly_returns)

    return check_retirement(
        strategy=strategy,
        rolling_6m_sharpe=sharpe_6m,
        consecutive_loss_months=consec_losses,
        rolling_12m_sharpe=sharpe_12m,
        current_dd=current_dd,
        backtest_max_dd=backtest_max_dd,
    )
