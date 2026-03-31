"""Seasonal momentum — computes forward return expectations based on historical same-period performance."""
import numpy as np


def seasonal_momentum(closes: np.ndarray, datetimes: np.ndarray,
                      lookback_years: int = 3) -> tuple:
    """Forward seasonal expectation based on historical same-period returns.

    For each bar, looks at the same calendar period in prior years and
    computes the average 5-day and 20-day forward returns that historically
    occurred at this time of year.

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
    expected_5d_return : np.ndarray
        Average historical 5-day forward return for this day-of-year.
    expected_20d_return : np.ndarray
        Average historical 20-day forward return for this day-of-year.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    exp_5d = np.full(n, np.nan)
    exp_20d = np.full(n, np.nan)

    # Day of year
    year_starts = datetimes.astype('datetime64[Y]')
    doy = (datetimes.astype('datetime64[D]') - year_starts).astype(int) + 1
    years = year_starts.astype(int) + 1970

    # Precompute forward returns (historical, so only used for past bars)
    fwd_5 = np.full(n, np.nan)
    fwd_20 = np.full(n, np.nan)
    if n > 5:
        fwd_5[:n - 5] = closes[5:] / closes[:n - 5] - 1.0
    if n > 20:
        fwd_20[:n - 20] = closes[20:] / closes[:n - 20] - 1.0

    # For each bar, find same day-of-year in prior years (using historical
    # forward returns only -- no look-ahead bias since we only use years < current)
    for i in range(252, n):
        cur_doy = doy[i]
        cur_year = years[i]

        # Match bars with similar day-of-year in prior years
        # Allow +/- 5 day tolerance for trading calendar differences
        mask = ((np.abs(doy[:i] - cur_doy) <= 5)
                & (years[:i] < cur_year)
                & (years[:i] >= cur_year - lookback_years))

        matched_5 = fwd_5[:i][mask]
        matched_20 = fwd_20[:i][mask]

        valid_5 = matched_5[np.isfinite(matched_5)]
        valid_20 = matched_20[np.isfinite(matched_20)]

        if len(valid_5) >= 2:
            exp_5d[i] = np.mean(valid_5)
        if len(valid_20) >= 2:
            exp_20d[i] = np.mean(valid_20)

    return exp_5d, exp_20d
