import numpy as np


def frama(closes: np.ndarray, period: int = 16) -> np.ndarray:
    """Fractal Adaptive Moving Average — adapts based on fractal dimension.

    Computes fractal dimension D from high-low range over two halves
    of the lookback window, then alpha = exp(-4.6 * (D - 1)).
    Faster adaptation in trending, slower in choppy markets.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    # Period must be even
    half = period // 2
    if n <= period:
        return np.full(n, np.nan)

    out = np.full(n, np.nan)
    out[period - 1] = closes[period - 1]

    for i in range(period, n):
        # First half: [i - period + 1 ... i - half]
        w1 = closes[i - period + 1 : i - half + 1]
        # Second half: [i - half + 1 ... i]
        w2 = closes[i - half + 1 : i + 1]
        # Full window
        w_full = closes[i - period + 1 : i + 1]

        hl1 = np.max(w1) - np.min(w1)
        hl2 = np.max(w2) - np.min(w2)
        hl_full = np.max(w_full) - np.min(w_full)

        # Fractal dimension
        if hl1 + hl2 > 1e-12 and hl_full > 1e-12:
            d = (np.log(hl1 + hl2) - np.log(hl_full)) / np.log(2.0)
        else:
            d = 1.0

        # Alpha from fractal dimension
        alpha = np.exp(-4.6 * (d - 1.0))
        alpha = max(0.01, min(1.0, alpha))

        prev = out[i - 1] if not np.isnan(out[i - 1]) else closes[i - 1]
        out[i] = alpha * closes[i] + (1.0 - alpha) * prev

    return out
