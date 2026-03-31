"""RSI of the price ratio — mean reversion signal on relative value.

When cross-asset RSI is extreme (>70 or <30), the ratio is likely
stretched and may revert.
"""

import numpy as np


def cross_asset_rsi(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """RSI applied to the A/B price ratio.

    Parameters
    ----------
    closes_a : closing prices of asset A.
    closes_b : closing prices of asset B.
    period   : RSI lookback period.

    Returns
    -------
    cross_rsi : array (0-100).  >70 = A overbought vs B, <30 = A oversold vs B.
    """
    n = len(closes_a)
    if n == 0:
        return np.array([], dtype=float)

    safe_b = np.where(closes_b == 0, np.nan, closes_b)
    ratio = closes_a / safe_b

    cross_rsi = np.full(n, np.nan)

    # Compute ratio changes
    delta = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ratio[i]) and not np.isnan(ratio[i - 1]):
            delta[i] = ratio[i] - ratio[i - 1]

    # Wilder smoothing for gains and losses
    avg_gain = 0.0
    avg_loss = 0.0
    count = 0

    for i in range(1, n):
        if np.isnan(delta[i]):
            continue
        gain = max(delta[i], 0.0)
        loss = max(-delta[i], 0.0)
        count += 1

        if count < period:
            avg_gain += gain
            avg_loss += loss
        elif count == period:
            avg_gain = (avg_gain + gain) / period
            avg_loss = (avg_loss + loss) / period
            if avg_loss == 0:
                cross_rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                cross_rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        else:
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            if avg_loss == 0:
                cross_rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                cross_rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return cross_rsi
