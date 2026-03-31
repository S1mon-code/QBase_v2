"""Month-of-year encoding — encodes calendar month as cyclical sin/cos features."""
import numpy as np


def month_cycle(datetimes: np.ndarray) -> tuple:
    """Month-of-year as cyclical sin/cos features.

    Encodes the month (1-12) as continuous cyclical features so that
    December and January are close together in feature space.

    Parameters
    ----------
    datetimes : np.ndarray
        numpy datetime64 array.

    Returns
    -------
    month_sin : np.ndarray
        sin(2*pi*(month-1)/12).
    month_cos : np.ndarray
        cos(2*pi*(month-1)/12).
    """
    n = len(datetimes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    months = (datetimes.astype('datetime64[M]')
              - datetimes.astype('datetime64[Y]')).astype(int) + 1

    phase = 2.0 * np.pi * (months.astype(float) - 1.0) / 12.0

    month_sin = np.sin(phase)
    month_cos = np.cos(phase)

    return month_sin, month_cos
