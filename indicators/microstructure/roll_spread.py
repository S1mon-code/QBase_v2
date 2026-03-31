"""Roll (1984) bid-ask spread estimator.

Estimates the effective bid-ask spread from the serial covariance of
consecutive price changes.  Works on trade/close data without needing
actual bid/ask quotes.
"""

import numpy as np


def roll_spread_estimate(
    closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll bid-ask spread estimate from serial covariance of returns.

    The Roll measure relies on the insight that a negative serial
    covariance in price changes is induced by bid-ask bounce.  The
    spread is estimated as 2 * sqrt(-cov) when cov < 0.

    Parameters
    ----------
    closes : closing prices.
    period : rolling window for covariance estimation.

    Returns
    -------
    (spread_estimate, spread_pct)
        spread_estimate – estimated absolute bid-ask spread.
        spread_pct      – spread as a percentage of the price.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if n < 3:
        return np.full(n, np.nan), np.full(n, np.nan)

    # Price changes
    dp = np.full(n, np.nan)
    dp[1:] = closes[1:] - closes[:-1]

    spread_est = np.full(n, np.nan)
    spread_pct = np.full(n, np.nan)

    for i in range(period + 1, n):
        # Serial covariance: cov(dp_t, dp_{t-1})
        dp_cur = dp[i - period + 1 : i + 1]
        dp_prev = dp[i - period : i]
        mask = ~(np.isnan(dp_cur) | np.isnan(dp_prev))
        if np.sum(mask) < 5:
            continue

        cov = np.mean(dp_cur[mask] * dp_prev[mask]) - np.mean(dp_cur[mask]) * np.mean(dp_prev[mask])

        if cov < 0:
            s = 2.0 * np.sqrt(-cov)
        else:
            # Positive serial covariance means no bid-ask bounce detected
            s = 0.0

        spread_est[i] = s
        if closes[i] != 0 and not np.isnan(closes[i]):
            spread_pct[i] = s / closes[i] * 100.0

    return spread_est, spread_pct
