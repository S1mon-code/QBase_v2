import numpy as np


def oi_volume_divergence(oi: np.ndarray, volumes: np.ndarray,
                         period: int = 20) -> tuple:
    """Divergence between OI change and volume.

    If OI changes significantly but volume is low, positions are
    being transferred (not newly created).  If OI change is small
    but volume is high, positions are being churned.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volume.
    period : int
        Lookback for normalisation.

    Returns
    -------
    divergence : np.ndarray
        Normalised divergence score.  Positive = OI change exceeds
        volume activity; negative = volume exceeds OI change.
    is_transfer : np.ndarray (float)
        1.0 when OI changes a lot but volume is low → position transfer.
    is_new_position : np.ndarray (float)
        1.0 when both OI change and volume are high → new positions.
    """
    n = len(oi)
    divergence = np.full(n, np.nan)
    is_transfer = np.zeros(n, dtype=float)
    is_new_position = np.zeros(n, dtype=float)

    if n < period + 1:
        return divergence, is_transfer, is_new_position

    abs_oi_chg = np.full(n, np.nan)
    abs_oi_chg[1:] = np.abs(np.diff(oi))

    for i in range(period + 1, n):
        oi_win = abs_oi_chg[i - period:i]
        vol_win = volumes[i - period:i]

        oi_mean = np.mean(oi_win)
        oi_std = np.std(oi_win, ddof=1)
        vol_mean = np.mean(vol_win)
        vol_std = np.std(vol_win, ddof=1)

        oi_z = (abs_oi_chg[i] - oi_mean) / oi_std if oi_std > 0 else 0.0
        vol_z = (volumes[i] - vol_mean) / vol_std if vol_std > 0 else 0.0

        divergence[i] = oi_z - vol_z

        # Transfer: high OI change, low volume
        if oi_z > 1.0 and vol_z < 0.0:
            is_transfer[i] = 1.0

        # New positions: both high
        if oi_z > 1.0 and vol_z > 1.0:
            is_new_position[i] = 1.0

    return divergence, is_transfer, is_new_position
