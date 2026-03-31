import numpy as np


def price_inertia(closes: np.ndarray, period: int = 20) -> tuple:
    """Price inertia: tendency to continue moving in same direction.

    Measures the autocorrelation of returns at lag 1 over a rolling window.
    High positive inertia = trending. Negative = mean reverting.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    period : int
        Rolling window for inertia measurement.

    Returns
    -------
    inertia : np.ndarray
        Rolling lag-1 autocorrelation of returns (-1 to 1).
    inertia_zscore : np.ndarray
        Z-score of inertia vs its own history.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    inertia = np.full(n, np.nan)
    inertia_z = np.full(n, np.nan)

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    for i in range(period + 1, n):
        window = rets[i - period:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < period // 2:
            continue

        # Lag-1 autocorrelation
        x = valid[:-1]
        y = valid[1:]
        if len(x) < 5:
            continue

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_std = np.std(x)
        y_std = np.std(y)

        if x_std > 1e-12 and y_std > 1e-12:
            corr = np.mean((x - x_mean) * (y - y_mean)) / (x_std * y_std)
            inertia[i] = np.clip(corr, -1.0, 1.0)

    # Z-score
    lookback = period * 5
    for i in range(period + 1 + lookback, n):
        hist = inertia[i - lookback:i + 1]
        valid = hist[np.isfinite(hist)]
        if len(valid) < lookback // 2:
            continue
        mu = np.mean(valid)
        std = np.std(valid)
        if std > 1e-9 and np.isfinite(inertia[i]):
            inertia_z[i] = (inertia[i] - mu) / std

    return inertia, inertia_z
