import numpy as np


def rolling_kurtosis(
    returns_or_closes: np.ndarray,
    period: int = 60,
) -> np.ndarray:
    """Rolling excess kurtosis of returns.

    If input looks like prices (all positive), log returns are computed.
    High kurtosis = fat tails (more extreme moves than normal distribution).
    Excess kurtosis: normal distribution = 0.
    Returns kurtosis array with NaN warmup.
    """
    n = len(returns_or_closes)
    if n < period:
        return np.full(n, np.nan)

    data = returns_or_closes.astype(np.float64)

    # Detect if input is prices or returns
    if np.all(data[~np.isnan(data)] > 0):
        rets = np.empty(n)
        rets[0] = np.nan
        rets[1:] = np.log(data[1:] / data[:-1])
    else:
        rets = data.copy()

    kurt = np.full(n, np.nan)
    for i in range(period, n):
        window = rets[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 4:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma == 0:
            kurt[i] = 0.0
            continue
        m4 = np.mean((valid - mu) ** 4)
        kurt[i] = m4 / (sigma ** 4) - 3.0  # excess kurtosis

    return kurt
