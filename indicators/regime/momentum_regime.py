import numpy as np


def momentum_regime(
    closes: np.ndarray,
    fast: int = 10,
    slow: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Momentum regime: accelerating, decelerating, reversing.

    Computes fast and slow rate-of-change, then classifies the
    momentum state based on their relationship and direction of change.

    Returns (regime, momentum_score).
    regime: 1=accelerating, 0=decelerating, -1=reversing.
    momentum_score: normalized momentum value.
    """
    n = len(closes)
    regime = np.full(n, np.nan, dtype=np.float64)
    momentum_score = np.full(n, np.nan, dtype=np.float64)

    if n < slow + 2:
        return regime, momentum_score

    safe = np.maximum(closes, 1e-9)

    # Fast and slow ROC
    fast_roc = np.full(n, np.nan, dtype=np.float64)
    slow_roc = np.full(n, np.nan, dtype=np.float64)

    for i in range(fast, n):
        fast_roc[i] = (safe[i] - safe[i - fast]) / safe[i - fast]

    for i in range(slow, n):
        slow_roc[i] = (safe[i] - safe[i - slow]) / safe[i - slow]

    # Classify momentum regime
    for i in range(slow + 1, n):
        fr = fast_roc[i]
        sr = slow_roc[i]
        fr_prev = fast_roc[i - 1]

        if np.isnan(fr) or np.isnan(sr) or np.isnan(fr_prev):
            continue

        # Momentum score: blend of fast and slow
        momentum_score[i] = 0.6 * fr + 0.4 * sr

        # Delta of fast ROC (acceleration)
        accel = fr - fr_prev

        # Regime classification
        if fr * sr > 0:
            # Same direction
            if abs(fr) > abs(sr) and accel * fr > 0:
                regime[i] = 1.0   # accelerating
            else:
                regime[i] = 0.0   # decelerating
        else:
            # Opposite direction: reversal
            regime[i] = -1.0

    return regime, momentum_score
