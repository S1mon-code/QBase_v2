"""Year progress encoding — encodes position within the year as cyclical sin/cos features."""
import numpy as np


def year_cycle(datetimes: np.ndarray) -> tuple:
    """Year progress as cyclical features using sin/cos encoding.

    Encodes the position within the year as continuous cyclical features,
    avoiding the discontinuity at year boundaries.

    Parameters
    ----------
    datetimes : np.ndarray
        numpy datetime64 array.

    Returns
    -------
    year_sin : np.ndarray
        sin(2*pi*day_of_year / days_in_year). Captures annual seasonality.
    year_cos : np.ndarray
        cos(2*pi*day_of_year / days_in_year). Together with sin, encodes
        full cyclical position.
    """
    n = len(datetimes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Day of year (1-based)
    year_starts = datetimes.astype('datetime64[Y]')
    day_of_year = (datetimes.astype('datetime64[D]') - year_starts).astype(int) + 1

    # Days in year (365 or 366)
    years = year_starts.astype(int) + 1970
    days_in_year = np.where(
        ((years % 4 == 0) & (years % 100 != 0)) | (years % 400 == 0),
        366.0, 365.0
    )

    phase = 2.0 * np.pi * day_of_year.astype(float) / days_in_year

    year_sin = np.sin(phase)
    year_cos = np.cos(phase)

    return year_sin, year_cos
