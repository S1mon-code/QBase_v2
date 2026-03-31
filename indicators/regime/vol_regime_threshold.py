import numpy as np


def vol_regime_simple(closes: np.ndarray, period: int = 60) -> tuple:
    """Simple 2-state volatility regime (high/low) using threshold switching.

    Uses a rolling volatility measure and classifies into high/low vol
    regimes based on the median. Also estimates transition probabilities.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    period : int
        Rolling window for volatility and regime calculation.

    Returns
    -------
    regime : np.ndarray
        0 = low vol regime, 1 = high vol regime.
    transition_prob : np.ndarray
        Estimated probability of switching to the other regime.
    vol_level : np.ndarray
        Current rolling volatility level (annualized).
    """
    n = len(closes)
    if n == 0:
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=float))

    regime = np.full(n, np.nan)
    trans_prob = np.full(n, np.nan)
    vol_level = np.full(n, np.nan)

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    # Rolling volatility
    vol_period = min(20, period // 3)
    for i in range(vol_period, n):
        window = rets[i - vol_period + 1:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) >= vol_period // 2:
            vol_level[i] = np.std(valid) * np.sqrt(252)

    # Regime classification using rolling median as threshold
    for i in range(period, n):
        vol_window = vol_level[i - period:i + 1]
        valid = vol_window[np.isfinite(vol_window)]
        if len(valid) < period // 2:
            continue

        median_vol = np.median(valid)
        if np.isfinite(vol_level[i]):
            regime[i] = 1.0 if vol_level[i] > median_vol else 0.0

    # Transition probabilities
    trans_window = period
    for i in range(period + trans_window, n):
        r_window = regime[i - trans_window:i]
        valid_r = r_window[np.isfinite(r_window)]
        if len(valid_r) < trans_window // 2:
            continue

        # Count transitions
        transitions = 0
        total = 0
        for j in range(1, len(valid_r)):
            if valid_r[j] != valid_r[j - 1]:
                transitions += 1
            total += 1

        if total > 0:
            trans_prob[i] = transitions / total

    return regime, trans_prob, vol_level
