import numpy as np


def instantaneous_trendline(closes: np.ndarray, alpha: float = 0.07) -> np.ndarray:
    """Ehlers Instantaneous Trendline — 2-pole IIR filter with near-zero lag.

    A low-lag trend filter that closely tracks price while removing
    high-frequency noise. Crossovers with price indicate trend changes.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n < 7:
        return np.full(n, np.nan)

    it = np.full(n, np.nan)
    it[0] = closes[0]
    it[1] = closes[1]
    it[2] = closes[2]
    it[3] = closes[3]
    it[4] = closes[4]
    it[5] = closes[5]
    it[6] = closes[6]

    # Seed with simple average
    for i in range(7):
        it[i] = (closes[i] + 2.0 * closes[max(0, i - 1)] + closes[max(0, i - 2)]) / 4.0

    for i in range(7, n):
        it[i] = (
            (alpha - alpha * alpha / 4.0) * closes[i]
            + 0.5 * alpha * alpha * closes[i - 1]
            - (alpha - 0.75 * alpha * alpha) * closes[i - 2]
            + 2.0 * (1.0 - alpha) * it[i - 1]
            - (1.0 - alpha) * (1.0 - alpha) * it[i - 2]
        )

    return it
