"""Monthly seasonal pattern — scores current month based on historical same-month return averages."""
import numpy as np


def monthly_seasonal(closes: np.ndarray, datetimes: np.ndarray,
                     lookback_years: int = 3) -> np.ndarray:
    """Monthly seasonal score based on historical same-month returns.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    datetimes : np.ndarray
        numpy datetime64 array.
    lookback_years : int
        Number of past years to average over.

    Returns
    -------
    seasonal_score : np.ndarray
        Range approximately -1 to +1. Positive = historically bullish month.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float)

    seasonal_score = np.full(n, np.nan)

    months = (datetimes.astype('datetime64[M]')
              - datetimes.astype('datetime64[Y]')).astype(int) + 1
    years = datetimes.astype('datetime64[Y]').astype(int) + 1970

    # Daily returns
    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    # Build monthly return history: for each bar, look back at same-month
    # returns from previous years and compute average.
    for i in range(1, n):
        cur_month = months[i]
        cur_year = years[i]

        mask = ((months[:i] == cur_month)
                & (years[:i] < cur_year)
                & (years[:i] >= cur_year - lookback_years)
                & np.isfinite(rets[:i]))

        if np.sum(mask) < 5:
            continue

        avg_ret = np.mean(rets[:i][mask])
        std_ret = np.std(rets[:i][mask])
        if std_ret == 0.0:
            seasonal_score[i] = 0.0
        else:
            raw = avg_ret / std_ret
            seasonal_score[i] = np.clip(raw, -1.0, 1.0)

    return seasonal_score
