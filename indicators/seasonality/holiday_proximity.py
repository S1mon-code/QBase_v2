"""Holiday proximity effect — scores pre/post-holiday return patterns for Chinese futures markets."""
import numpy as np


def holiday_effect(closes: np.ndarray, datetimes: np.ndarray,
                   lookback: int = 252) -> np.ndarray:
    """Pre/post holiday return patterns for Chinese futures markets.

    Detects proximity to major Chinese holidays (Spring Festival ~Feb,
    National Day ~Oct) and scores based on historical pre/post-holiday returns.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    datetimes : np.ndarray
        numpy datetime64 array.
    lookback : int
        Rolling lookback window for computing historical patterns.

    Returns
    -------
    holiday_score : np.ndarray
        Positive before historically bullish holidays, negative otherwise.
        Range approximately -1 to +1.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float)

    holiday_score = np.full(n, np.nan)

    months = (datetimes.astype('datetime64[M]')
              - datetimes.astype('datetime64[Y]')).astype(int) + 1
    days = (datetimes.astype('datetime64[D]')
            - datetimes.astype('datetime64[M]')).astype(int) + 1

    # Holiday proximity windows (month, day_range_start, day_range_end)
    # Spring Festival: typically late Jan to mid Feb
    # National Day: Oct 1-7
    # May Day: May 1-5
    # These are approximate; actual dates shift each year for lunar calendar.
    holiday_windows = [
        (1, 20, 31),   # Pre-Spring Festival (late Jan)
        (2, 1, 15),    # Post-Spring Festival (early Feb)
        (4, 25, 30),   # Pre-May Day
        (5, 1, 10),    # Post-May Day
        (9, 20, 30),   # Pre-National Day
        (10, 1, 15),   # Post-National Day
    ]

    # Tag each bar as near-holiday or not
    near_holiday = np.zeros(n, dtype=bool)
    for m, d_start, d_end in holiday_windows:
        near_holiday |= (months == m) & (days >= d_start) & (days <= d_end)

    # Daily returns
    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    for i in range(60, n):
        start = max(1, i - lookback)

        hol_mask = near_holiday[start:i] & np.isfinite(rets[start:i])
        non_mask = (~near_holiday[start:i]) & np.isfinite(rets[start:i])

        hol_rets = rets[start:i][hol_mask]
        non_rets = rets[start:i][non_mask]

        if len(hol_rets) < 3 or len(non_rets) < 5:
            continue

        diff = np.mean(hol_rets) - np.mean(non_rets)
        pooled_std = np.sqrt((np.var(hol_rets) + np.var(non_rets)) / 2.0)

        if near_holiday[i] and pooled_std > 0:
            holiday_score[i] = np.clip(diff / pooled_std, -1.0, 1.0)
        else:
            holiday_score[i] = 0.0

    return holiday_score
