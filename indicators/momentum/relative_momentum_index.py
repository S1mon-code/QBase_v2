import numpy as np


def rmi(
    closes: np.ndarray,
    period: int = 14,
    lookback: int = 5,
) -> np.ndarray:
    """Relative Momentum Index — RSI variant comparing close to close[lookback] bars ago.

    Instead of comparing close to close[1] like RSI, RMI compares
    close[i] to close[i - lookback]. Range 0-100.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    warmup = lookback + period
    if n <= warmup:
        return np.full(n, np.nan)

    # Momentum differences (close[i] - close[i - lookback])
    deltas = np.full(n, np.nan)
    deltas[lookback:] = closes[lookback:] - closes[:-lookback]

    gains = np.where(np.nan_to_num(deltas, nan=0.0) > 0, deltas, 0.0)
    losses = np.where(np.nan_to_num(deltas, nan=0.0) < 0, -deltas, 0.0)

    # Wilder smoothing seeded from lookback onward
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    start = lookback + period
    avg_gain[start] = np.mean(gains[lookback + 1 : start + 1])
    avg_loss[start] = np.mean(losses[lookback + 1 : start + 1])

    for i in range(start + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    out = np.full(n, np.nan)
    for i in range(start, n):
        if np.isnan(avg_gain[i]) or np.isnan(avg_loss[i]):
            continue
        if avg_loss[i] < 1e-12:
            out[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            out[i] = 100.0 - 100.0 / (1.0 + rs)

    return out
