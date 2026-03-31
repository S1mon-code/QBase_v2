import numpy as np


def oi_stress(closes: np.ndarray, oi: np.ndarray, volumes: np.ndarray,
              period: int = 20) -> tuple:
    """Market stress indicator from OI.

    Stress occurs when high OI + high volume + large price moves
    coincide — indicating forced liquidation, margin calls, or
    panic positioning.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volume.
    period : int
        Lookback for normalisation.

    Returns
    -------
    stress_score : np.ndarray
        Composite stress score (0 = calm, higher = more stress).
    is_stress : np.ndarray (float)
        1.0 when stress_score exceeds 2 standard deviations, else 0.0.
    """
    n = len(closes)
    stress_score = np.full(n, np.nan)
    is_stress = np.zeros(n, dtype=float)

    if n < period + 1:
        return stress_score, is_stress

    abs_ret = np.full(n, np.nan)
    abs_ret[1:] = np.abs(np.diff(closes) / closes[:-1])

    for i in range(period + 1, n):
        # Z-score of |return|
        ret_win = abs_ret[i - period:i]
        ret_m = np.mean(ret_win)
        ret_s = np.std(ret_win, ddof=1)
        ret_z = (abs_ret[i] - ret_m) / ret_s if ret_s > 0 else 0.0

        # Z-score of volume
        vol_win = volumes[i - period:i]
        vol_m = np.mean(vol_win)
        vol_s = np.std(vol_win, ddof=1)
        vol_z = (volumes[i] - vol_m) / vol_s if vol_s > 0 else 0.0

        # Z-score of OI level
        oi_win = oi[i - period:i]
        oi_m = np.mean(oi_win)
        oi_s = np.std(oi_win, ddof=1)
        oi_z = (oi[i] - oi_m) / oi_s if oi_s > 0 else 0.0

        # Stress = product of extremes (all positive contributions)
        score = abs(ret_z) * max(0, vol_z) * max(0, abs(oi_z))
        # Take cube root to keep scale manageable
        stress_score[i] = np.cbrt(score)

    # Determine stress threshold
    valid = stress_score[~np.isnan(stress_score)]
    if len(valid) > 10:
        threshold = np.mean(valid) + 2.0 * np.std(valid)
        for i in range(n):
            if not np.isnan(stress_score[i]) and stress_score[i] > threshold:
                is_stress[i] = 1.0

    return stress_score, is_stress
