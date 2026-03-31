import numpy as np


def relative_volatility(
    closes: np.ndarray,
    fast: int = 10,
    slow: int = 60,
) -> tuple:
    """Relative Volatility = fast_vol / slow_vol.

    >1 means volatility is expanding, <1 means contracting.
    Returns (rv, rv_zscore).
    """
    n = len(closes)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan)

    closes = closes.astype(np.float64)
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(closes[1:] / closes[:-1])

    fast_vol = np.full(n, np.nan)
    slow_vol = np.full(n, np.nan)

    for i in range(fast, n):
        fast_vol[i] = np.nanstd(log_ret[i - fast + 1 : i + 1], ddof=1)

    for i in range(slow, n):
        slow_vol[i] = np.nanstd(log_ret[i - slow + 1 : i + 1], ddof=1)

    rv = np.full(n, np.nan)
    valid = (~np.isnan(fast_vol)) & (~np.isnan(slow_vol)) & (slow_vol > 0)
    rv[valid] = fast_vol[valid] / slow_vol[valid]

    # Z-score of rv over slow window
    rv_zscore = np.full(n, np.nan)
    for i in range(slow, n):
        window = rv[i - slow + 1 : i + 1]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) >= 5:
            mu = np.mean(valid_w)
            sigma = np.std(valid_w, ddof=1)
            if sigma > 0:
                rv_zscore[i] = (rv[i] - mu) / sigma

    return rv, rv_zscore
