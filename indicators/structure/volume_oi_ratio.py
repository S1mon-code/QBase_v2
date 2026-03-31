import numpy as np


def volume_oi_ratio(volumes: np.ndarray, oi: np.ndarray,
                    period: int = 20) -> tuple:
    """Volume/OI ratio -- high ratio indicates high turnover/speculation.

    Parameters
    ----------
    volumes : np.ndarray
        Trading volumes.
    oi : np.ndarray
        Open interest.
    period : int
        Rolling period for z-score computation.

    Returns
    -------
    ratio : np.ndarray
        Volume / OI ratio.
    ratio_zscore : np.ndarray
        Z-score of the ratio over rolling window.
    is_speculative : np.ndarray (bool)
        True if ratio z-score > 2.0 (speculative regime).
    """
    n = len(volumes)
    if n == 0:
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=bool))

    ratio = np.full(n, np.nan)
    ratio_z = np.full(n, np.nan)
    is_spec = np.zeros(n, dtype=bool)

    # Compute ratio
    for i in range(n):
        if oi[i] > 0 and np.isfinite(volumes[i]) and np.isfinite(oi[i]):
            ratio[i] = volumes[i] / oi[i]

    # Rolling z-score
    for i in range(period, n):
        window = ratio[i - period:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < 5:
            continue

        mu = np.mean(valid)
        std = np.std(valid)
        if std > 0 and np.isfinite(ratio[i]):
            ratio_z[i] = (ratio[i] - mu) / std
            is_spec[i] = ratio_z[i] > 2.0

    return ratio, ratio_z, is_spec
