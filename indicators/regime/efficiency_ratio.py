import numpy as np


def efficiency_ratio(
    closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Kaufman efficiency ratio: net price change / sum of absolute changes.

    Measures how efficiently price moves from point A to point B.
    A value near 1.0 means price moved in a straight line (perfect trend),
    near 0.0 means price went nowhere despite lots of movement (noise).

    Returns (er, er_smoothed). 1.0=perfect trend, 0.0=pure noise.
    er_smoothed uses a 10-bar EMA of er for smoother signals.
    """
    n = len(closes)
    er = np.full(n, np.nan, dtype=np.float64)
    er_smoothed = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return er, er_smoothed

    for i in range(period, n):
        net_change = abs(closes[i] - closes[i - period])
        sum_changes = np.sum(np.abs(np.diff(closes[i - period : i + 1])))

        if sum_changes > 1e-14:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0

    # EMA smoothing
    smooth_period = 10
    alpha = 2.0 / (smooth_period + 1.0)

    first_valid = None
    for i in range(n):
        if not np.isnan(er[i]):
            first_valid = i
            break

    if first_valid is None:
        return er, er_smoothed

    er_smoothed[first_valid] = er[first_valid]
    for i in range(first_valid + 1, n):
        if np.isnan(er[i]):
            continue
        if np.isnan(er_smoothed[i - 1]):
            er_smoothed[i] = er[i]
        else:
            er_smoothed[i] = alpha * er[i] + (1.0 - alpha) * er_smoothed[i - 1]

    return er, er_smoothed
