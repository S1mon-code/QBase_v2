import numpy as np


def oi_climax(oi: np.ndarray, volumes: np.ndarray,
              period: int = 20, threshold: float = 2.0) -> tuple:
    """Detect OI climax events.

    A climax occurs when both OI change and volume are extreme
    simultaneously — indicating a potential exhaustion or
    capitulation event.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volume.
    period : int
        Lookback for z-score normalisation.
    threshold : float
        Z-score threshold for both OI change and volume to qualify.

    Returns
    -------
    is_climax : np.ndarray (float)
        1.0 on climax bars, 0.0 otherwise.
    climax_direction : np.ndarray
        +1.0 for OI surge climax (extreme buildup),
        -1.0 for OI collapse climax (extreme unwind), 0.0 otherwise.
    climax_strength : np.ndarray
        Product of OI-change z-score and volume z-score.
    """
    n = len(oi)
    is_climax = np.zeros(n, dtype=float)
    climax_direction = np.zeros(n, dtype=float)
    climax_strength = np.full(n, np.nan)

    if n < period + 1:
        return is_climax, climax_direction, climax_strength

    oi_change = np.full(n, np.nan)
    oi_change[1:] = np.diff(oi)

    for i in range(period + 1, n):
        oi_win = oi_change[i - period:i]
        vol_win = volumes[i - period:i]

        oi_m = np.mean(oi_win)
        oi_s = np.std(oi_win, ddof=1)
        vol_m = np.mean(vol_win)
        vol_s = np.std(vol_win, ddof=1)

        oi_z = (oi_change[i] - oi_m) / oi_s if oi_s > 0 else 0.0
        vol_z = (volumes[i] - vol_m) / vol_s if vol_s > 0 else 0.0

        strength = abs(oi_z) * abs(vol_z)
        climax_strength[i] = strength

        if abs(oi_z) >= threshold and abs(vol_z) >= threshold:
            is_climax[i] = 1.0
            climax_direction[i] = 1.0 if oi_z > 0 else -1.0

    return is_climax, climax_direction, climax_strength
