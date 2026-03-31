import numpy as np


def rolling_skewness(
    returns_or_closes: np.ndarray,
    period: int = 60,
) -> np.ndarray:
    """Rolling skewness of returns.

    If input looks like prices (all positive, no negatives), log returns are computed.
    Negative skew indicates crash risk (fat left tail).
    Returns skewness array with NaN warmup.
    """
    n = len(returns_or_closes)
    if n < period:
        return np.full(n, np.nan)

    data = returns_or_closes.astype(np.float64)

    # Detect if input is prices (all positive) or returns
    if np.all(data[~np.isnan(data)] > 0):
        # Treat as prices, compute log returns
        rets = np.empty(n)
        rets[0] = np.nan
        rets[1:] = np.log(data[1:] / data[:-1])
    else:
        rets = data.copy()

    skew = np.full(n, np.nan)
    for i in range(period, n):
        window = rets[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 3:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma == 0:
            skew[i] = 0.0
            continue
        m3 = np.mean((valid - mu) ** 3)
        skew[i] = m3 / (sigma ** 3)

    return skew
