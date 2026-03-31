"""Alpha Decay Detection.

Monitors rolling Information Coefficient (IC) to detect whether a strategy's
predictive power is declining over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class DecayResult:
    """Result of alpha decay analysis."""

    rolling_ic: np.ndarray  # rolling IC values
    ic_trend: float  # slope of IC over time (negative = decaying)
    is_decaying: bool  # True if trend significantly negative
    half_life_bars: float | None  # estimated half-life of alpha


def _rolling_spearman_ic(
    signals: np.ndarray,
    forward_returns: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute rolling Spearman rank correlation (IC).

    Args:
        signals: 1-D array of signal values.
        forward_returns: 1-D array of corresponding forward returns.
        window: Rolling window size.

    Returns:
        1-D array of rolling IC values (length = n - window + 1).
    """
    n = len(signals)
    ic_values: list[float] = []

    for i in range(n - window + 1):
        sig_window = signals[i : i + window]
        ret_window = forward_returns[i : i + window]

        # Skip windows with zero variance
        if np.std(sig_window) == 0.0 or np.std(ret_window) == 0.0:
            ic_values.append(0.0)
            continue

        corr, _ = scipy_stats.spearmanr(sig_window, ret_window)
        ic_values.append(float(corr) if np.isfinite(corr) else 0.0)

    return np.array(ic_values, dtype=np.float64)


def detect_alpha_decay(
    signals: np.ndarray,
    forward_returns: np.ndarray,
    window: int = 60,
    lookback: int = 252,
) -> DecayResult:
    """Detect alpha decay via rolling IC (Information Coefficient = rank correlation).

    Computes rolling Spearman rank correlation between signals and forward returns,
    then fits a linear trend to the IC series. Negative slope indicates decay.

    Args:
        signals: 1-D array of signal values.
        forward_returns: 1-D array of corresponding forward returns.
        window: Rolling window size for IC computation.
        lookback: Number of most recent IC observations to use for trend fitting.

    Returns:
        DecayResult with rolling IC, trend slope, decay flag, and half-life.

    Raises:
        ValueError: If arrays are empty or have mismatched lengths.
    """
    sig = np.asarray(signals, dtype=np.float64).ravel()
    fwd = np.asarray(forward_returns, dtype=np.float64).ravel()

    if len(sig) == 0:
        raise ValueError("signals must not be empty")
    if len(sig) != len(fwd):
        raise ValueError("signals and forward_returns must have the same length")
    if len(sig) < window:
        raise ValueError(
            f"signals length ({len(sig)}) must be >= window ({window})"
        )

    rolling_ic = _rolling_spearman_ic(sig, fwd, window)

    # Use only the most recent lookback observations for trend
    ic_for_trend = rolling_ic[-lookback:] if len(rolling_ic) > lookback else rolling_ic

    # Fit linear trend: IC = slope * t + intercept
    t = np.arange(len(ic_for_trend), dtype=np.float64)
    if len(ic_for_trend) < 2:
        ic_trend = 0.0
        p_value = 1.0
    else:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(t, ic_for_trend)
        ic_trend = float(slope)

    # Decay if slope is significantly negative (p < 0.05)
    is_decaying = ic_trend < 0.0 and p_value < 0.05

    # Estimate half-life: time for IC to halve from current level
    half_life_bars: float | None = None
    if is_decaying and ic_trend < 0.0:
        current_ic = float(ic_for_trend[-1]) if len(ic_for_trend) > 0 else 0.0
        if current_ic > 0.0:
            # bars until IC drops to half current level
            half_life_bars = abs(current_ic / 2.0 / ic_trend)

    return DecayResult(
        rolling_ic=rolling_ic,
        ic_trend=ic_trend,
        is_decaying=is_decaying,
        half_life_bars=half_life_bars,
    )
