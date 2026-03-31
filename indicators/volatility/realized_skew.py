import numpy as np


def realized_skewness(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Realized skewness from intraday proxies (Barndorff-Nielsen style).

    Uses high-low range as a proxy for intraday variation to compute
    a realized skewness measure. Negative values indicate asymmetric
    downside risk.
    Returns rskew array with NaN warmup.
    """
    n = len(closes)
    if n < 2:
        return np.full(n, np.nan)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)

    # Log return and realized variance proxy
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(closes[1:] / closes[:-1])

    # Parkinson variance proxy per bar
    log_hl = np.log(highs / lows)
    rv_proxy = log_hl ** 2 / (4.0 * np.log(2.0))

    rskew = np.full(n, np.nan)
    for i in range(period, n):
        rets = log_ret[i - period + 1 : i + 1]
        rvs = rv_proxy[i - period + 1 : i + 1]

        valid_mask = ~np.isnan(rets) & ~np.isnan(rvs) & (rvs > 0)
        if np.sum(valid_mask) < 5:
            continue

        r = rets[valid_mask]
        v = rvs[valid_mask]

        # Realized skewness: sum(r^3) / (sum(r^2))^1.5 * sqrt(n)
        sum_r2 = np.sum(r ** 2)
        if sum_r2 == 0:
            continue
        sum_r3 = np.sum(r ** 3)
        nn = len(r)
        rskew[i] = np.sqrt(nn) * sum_r3 / (sum_r2 ** 1.5)

    return rskew
