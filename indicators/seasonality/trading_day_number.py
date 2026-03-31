"""Trading day number — counts sequential trading day within each month and flags first/last five days."""
import numpy as np


def trading_day_of_month(datetimes: np.ndarray) -> tuple:
    """Trading day number within the month.

    Counts the sequential trading day within each month (1st trading day,
    2nd trading day, etc.) and also flags the first/last 5 trading days.

    Parameters
    ----------
    datetimes : np.ndarray
        numpy datetime64 array. Assumed to contain only trading days
        (non-trading days already excluded).

    Returns
    -------
    day_number : np.ndarray (int)
        Trading day number within the month (1, 2, 3, ...).
    is_first_5_days : np.ndarray (bool)
        True if within first 5 trading days of the month.
    is_last_5_days : np.ndarray (bool)
        True if within last 5 trading days of the month.
    """
    n = len(datetimes)
    if n == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=bool),
                np.array([], dtype=bool))

    year_months = datetimes.astype('datetime64[M]')

    day_number = np.ones(n, dtype=int)

    # Forward pass: count trading day within month
    for i in range(1, n):
        if year_months[i] == year_months[i - 1]:
            day_number[i] = day_number[i - 1] + 1
        else:
            day_number[i] = 1

    # Reverse pass: count days from end of month
    days_from_end = np.ones(n, dtype=int)
    for i in range(n - 2, -1, -1):
        if year_months[i] == year_months[i + 1]:
            days_from_end[i] = days_from_end[i + 1] + 1
        else:
            days_from_end[i] = 1

    is_first_5_days = day_number <= 5
    is_last_5_days = days_from_end <= 5

    return day_number, is_first_5_days, is_last_5_days
