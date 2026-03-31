"""Rate of change of price ratio — captures changing relative value.

When the ratio ROC is accelerating, one asset is gaining relative
strength versus the other at an increasing pace.
"""

import numpy as np


def ratio_momentum(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    period: int = 20,
    lookback: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Momentum (rate of change) of the A/B price ratio.

    Parameters
    ----------
    closes_a : closing prices of asset A.
    closes_b : closing prices of asset B.
    period   : ROC lookback period.
    lookback : smoothing window for signal line.

    Returns
    -------
    (ratio_roc, ratio_roc_signal)
        ratio_roc        – rate of change of A/B ratio.
        ratio_roc_signal – smoothed signal line (SMA of ratio_roc).
    """
    n = len(closes_a)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    safe_b = np.where(closes_b == 0, np.nan, closes_b)
    ratio = closes_a / safe_b

    roc = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(ratio[i]) and not np.isnan(ratio[i - period]):
            if ratio[i - period] != 0:
                roc[i] = (ratio[i] / ratio[i - period] - 1.0) * 100.0

    signal = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        window = roc[i - lookback + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            signal[i] = np.mean(valid)

    return roc, signal
