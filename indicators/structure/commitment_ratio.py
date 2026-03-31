import numpy as np


def commitment_ratio(oi: np.ndarray, volumes: np.ndarray,
                     period: int = 20) -> tuple:
    """OI/Volume as commitment level.

    High ratio means participants hold positions rather than day-trade,
    indicating committed holders and potentially stronger trends.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volumes.
    period : int
        Rolling window for percentile calculation.

    Returns
    -------
    ratio : np.ndarray
        OI / Volume ratio.
    ratio_percentile : np.ndarray
        Rolling percentile rank (0-1) of the ratio.
    """
    n = len(oi)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    ratio = np.full(n, np.nan)
    ratio_pctl = np.full(n, np.nan)

    for i in range(n):
        if (np.isfinite(oi[i]) and np.isfinite(volumes[i])
                and volumes[i] > 0):
            ratio[i] = oi[i] / volumes[i]

    # Rolling percentile
    for i in range(period, n):
        window = ratio[i - period:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < period // 2:
            continue
        if np.isfinite(ratio[i]):
            ratio_pctl[i] = np.sum(valid <= ratio[i]) / len(valid)

    return ratio, ratio_pctl
