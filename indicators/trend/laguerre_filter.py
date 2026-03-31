import numpy as np


def laguerre(closes: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """Laguerre filter — 4-element Laguerre transform for smooth trend.

    Uses a cascade of 4 first-order IIR filters with damping factor gamma.
    The output is the average of the 4 Laguerre elements, producing a
    very smooth trend line. Higher gamma = more smoothing, more lag.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n < 2:
        return np.full(n, np.nan)

    L0 = np.zeros(n)
    L1 = np.zeros(n)
    L2 = np.zeros(n)
    L3 = np.zeros(n)
    out = np.full(n, np.nan)

    L0[0] = closes[0]
    L1[0] = closes[0]
    L2[0] = closes[0]
    L3[0] = closes[0]

    for i in range(1, n):
        L0[i] = (1.0 - gamma) * closes[i] + gamma * L0[i - 1]
        L1[i] = -gamma * L0[i] + L0[i - 1] + gamma * L1[i - 1]
        L2[i] = -gamma * L1[i] + L1[i - 1] + gamma * L2[i - 1]
        L3[i] = -gamma * L2[i] + L2[i - 1] + gamma * L3[i - 1]
        out[i] = (L0[i] + L1[i] + L2[i] + L3[i]) / 4.0

    # NaN warmup (first few bars are initializing)
    warmup = max(4, int(1.0 / (1.0 - gamma)))
    warmup = min(warmup, n)
    out[:warmup] = np.nan

    return out
