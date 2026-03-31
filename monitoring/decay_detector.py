"""Alpha Decay Detection.

Monitors rolling performance metrics to detect early signs of alpha decay
in live or paper trading strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecayAlert:
    """A single alpha-decay alert.

    Attributes
    ----------
    metric : str
        Which metric triggered the alert
        (``"rolling_sharpe"``, ``"backtest_deviation"``, ``"trade_frequency"``).
    current_value : float
        The observed value of the metric.
    threshold : float
        The threshold that was breached.
    level : str
        ``"yellow"`` or ``"red"``.
    message : str
        Human-readable description.
    """

    metric: str
    current_value: float
    threshold: float
    level: str
    message: str


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_rolling_sharpe(
    daily_returns: np.ndarray,
    window: int = 60,
    yellow_threshold: int = 10,
    red_threshold: int = 20,
) -> DecayAlert | None:
    """Check if rolling Sharpe has been below 0 for too many consecutive days.

    Parameters
    ----------
    daily_returns : np.ndarray
        1-D array of daily strategy returns.
    window : int
        Rolling window size in days for Sharpe calculation.
    yellow_threshold : int
        Consecutive days below 0 to trigger a yellow alert.
    red_threshold : int
        Consecutive days below 0 to trigger a red alert.

    Returns
    -------
    DecayAlert | None
        Alert if threshold breached, otherwise ``None``.
    """
    if len(daily_returns) < window:
        return None

    # Rolling Sharpe: annualised = mean/std * sqrt(252)
    rolling_mean = np.convolve(daily_returns, np.ones(window) / window, mode="valid")
    # rolling std via variance identity
    rolling_sq = np.convolve(daily_returns ** 2, np.ones(window) / window, mode="valid")
    rolling_var = rolling_sq - rolling_mean ** 2
    # Clamp tiny negative variances from floating-point noise
    rolling_var = np.maximum(rolling_var, 0.0)
    rolling_std = np.sqrt(rolling_var)

    # Avoid divide-by-zero: where std == 0, Sharpe is 0 (no information)
    safe_std = np.where(rolling_std == 0, 1.0, rolling_std)
    rolling_sharpe = (rolling_mean / safe_std) * np.sqrt(252)
    rolling_sharpe = np.where(rolling_std == 0, 0.0, rolling_sharpe)

    # Count consecutive days below 0 at the tail
    below_zero = rolling_sharpe < 0
    consecutive = 0
    for val in reversed(below_zero):
        if val:
            consecutive += 1
        else:
            break

    if consecutive >= red_threshold:
        return DecayAlert(
            metric="rolling_sharpe",
            current_value=float(consecutive),
            threshold=float(red_threshold),
            level="red",
            message=(
                f"Rolling {window}d Sharpe below 0 for {consecutive} consecutive "
                f"days (red threshold: {red_threshold})."
            ),
        )
    if consecutive >= yellow_threshold:
        return DecayAlert(
            metric="rolling_sharpe",
            current_value=float(consecutive),
            threshold=float(yellow_threshold),
            level="yellow",
            message=(
                f"Rolling {window}d Sharpe below 0 for {consecutive} consecutive "
                f"days (yellow threshold: {yellow_threshold})."
            ),
        )
    return None


def check_backtest_deviation(
    live_sharpe: float,
    backtest_sharpe: float,
    backtest_std: float,
    sigma_yellow: float = 1.5,
    sigma_red: float = 2.0,
) -> DecayAlert | None:
    """Check if live performance deviates from backtest.

    Parameters
    ----------
    live_sharpe : float
        Observed Sharpe in live / paper trading.
    backtest_sharpe : float
        Expected Sharpe from backtest.
    backtest_std : float
        Standard deviation of backtest Sharpe estimate.
    sigma_yellow : float
        Number of sigma for yellow alert.
    sigma_red : float
        Number of sigma for red alert.

    Returns
    -------
    DecayAlert | None
        Alert if deviation exceeds threshold, otherwise ``None``.
    """
    if backtest_std <= 0:
        return None

    deviation = (backtest_sharpe - live_sharpe) / backtest_std

    if deviation >= sigma_red:
        return DecayAlert(
            metric="backtest_deviation",
            current_value=deviation,
            threshold=sigma_red,
            level="red",
            message=(
                f"Live Sharpe ({live_sharpe:.2f}) deviates {deviation:.1f}σ below "
                f"backtest ({backtest_sharpe:.2f} ± {backtest_std:.2f}). "
                f"Red threshold: {sigma_red}σ."
            ),
        )
    if deviation >= sigma_yellow:
        return DecayAlert(
            metric="backtest_deviation",
            current_value=deviation,
            threshold=sigma_yellow,
            level="yellow",
            message=(
                f"Live Sharpe ({live_sharpe:.2f}) deviates {deviation:.1f}σ below "
                f"backtest ({backtest_sharpe:.2f} ± {backtest_std:.2f}). "
                f"Yellow threshold: {sigma_yellow}σ."
            ),
        )
    return None


def check_trade_frequency(
    actual_trades_per_month: float,
    expected_trades_per_month: float,
    deviation_yellow: float = 0.5,
    deviation_red: float = 1.0,
) -> DecayAlert | None:
    """Check if actual trading frequency deviates from expected.

    Parameters
    ----------
    actual_trades_per_month : float
        Observed trades per month.
    expected_trades_per_month : float
        Expected trades per month from backtest.
    deviation_yellow : float
        Fractional deviation for yellow alert (0.5 = 50%).
    deviation_red : float
        Fractional deviation for red alert (1.0 = 100%).

    Returns
    -------
    DecayAlert | None
        Alert if deviation exceeds threshold, otherwise ``None``.
    """
    if expected_trades_per_month <= 0:
        return None

    relative_deviation = abs(actual_trades_per_month - expected_trades_per_month) / expected_trades_per_month

    if relative_deviation >= deviation_red:
        return DecayAlert(
            metric="trade_frequency",
            current_value=relative_deviation,
            threshold=deviation_red,
            level="red",
            message=(
                f"Trade frequency deviation {relative_deviation:.0%} "
                f"(actual {actual_trades_per_month:.1f} vs expected "
                f"{expected_trades_per_month:.1f}/month). "
                f"Red threshold: {deviation_red:.0%}."
            ),
        )
    if relative_deviation >= deviation_yellow:
        return DecayAlert(
            metric="trade_frequency",
            current_value=relative_deviation,
            threshold=deviation_yellow,
            level="yellow",
            message=(
                f"Trade frequency deviation {relative_deviation:.0%} "
                f"(actual {actual_trades_per_month:.1f} vs expected "
                f"{expected_trades_per_month:.1f}/month). "
                f"Yellow threshold: {deviation_yellow:.0%}."
            ),
        )
    return None


# ---------------------------------------------------------------------------
# Aggregated runner
# ---------------------------------------------------------------------------

def run_all_checks(
    daily_returns: np.ndarray | None = None,
    live_sharpe: float | None = None,
    backtest_sharpe: float | None = None,
    backtest_std: float | None = None,
    actual_trades: float | None = None,
    expected_trades: float | None = None,
) -> list[DecayAlert]:
    """Run all available decay checks.

    Only runs checks for which the required data is provided.

    Returns
    -------
    list[DecayAlert]
        List of triggered alerts (empty means all clear).
    """
    alerts: list[DecayAlert] = []

    if daily_returns is not None:
        alert = check_rolling_sharpe(daily_returns)
        if alert is not None:
            alerts.append(alert)

    if live_sharpe is not None and backtest_sharpe is not None and backtest_std is not None:
        alert = check_backtest_deviation(live_sharpe, backtest_sharpe, backtest_std)
        if alert is not None:
            alerts.append(alert)

    if actual_trades is not None and expected_trades is not None:
        alert = check_trade_frequency(actual_trades, expected_trades)
        if alert is not None:
            alerts.append(alert)

    return alerts
