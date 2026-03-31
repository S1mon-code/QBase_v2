import numpy as np


def vidya(
    closes: np.ndarray,
    period: int = 14,
    cmo_period: int = 9,
) -> np.ndarray:
    """Variable Index Dynamic Average — adapts speed based on CMO ratio.

    VIDYA[i] = alpha * |CMO| * close[i] + (1 - alpha * |CMO|) * VIDYA[i-1]

    Where alpha = 2 / (period + 1) and CMO is the Chande Momentum Oscillator.
    Faster in trending markets, slower in choppy markets.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    warmup = max(period, cmo_period)
    if n <= warmup:
        return np.full(n, np.nan)

    alpha = 2.0 / (period + 1)

    # CMO calculation
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    cmo_abs = np.full(n, np.nan)
    for i in range(cmo_period, n):
        g = np.sum(gains[i - cmo_period + 1 : i + 1])
        l = np.sum(losses[i - cmo_period + 1 : i + 1])
        denom = g + l
        if denom > 1e-12:
            cmo_abs[i] = abs((g - l) / denom)
        else:
            cmo_abs[i] = 0.0

    # VIDYA
    out = np.full(n, np.nan)
    # Seed at warmup
    out[warmup] = closes[warmup]
    for i in range(warmup + 1, n):
        if np.isnan(cmo_abs[i]):
            out[i] = out[i - 1]
        else:
            sc = alpha * cmo_abs[i]
            out[i] = sc * closes[i] + (1.0 - sc) * out[i - 1]

    return out
