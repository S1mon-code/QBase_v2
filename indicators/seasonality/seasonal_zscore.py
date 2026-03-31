"""Seasonal z-score — measures current price deviation from historical seasonal expected levels."""
import numpy as np


def seasonal_zscore(closes: np.ndarray, datetimes: np.ndarray,
                    period: int = 252) -> tuple:
    """Price vs seasonal expected level z-score.

    For each bar, computes the expected price based on same-calendar-day
    historical returns and measures how far current price deviates.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    datetimes : np.ndarray
        numpy datetime64 array.
    period : int
        Approximate bars per year for seasonal cycle.

    Returns
    -------
    sz : np.ndarray
        Seasonal z-score: positive = above seasonal norm.
    seasonal_band_upper : np.ndarray
        Upper 1-sigma seasonal band.
    seasonal_band_lower : np.ndarray
        Lower 1-sigma seasonal band.
    """
    n = len(closes)
    if n == 0:
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=float))

    sz = np.full(n, np.nan)
    band_upper = np.full(n, np.nan)
    band_lower = np.full(n, np.nan)

    # Compute log returns
    log_prices = np.log(np.maximum(closes, 1e-9))

    # For each bar, compare with bars at multiples of ``period`` ago
    min_years = 2
    for i in range(min_years * period, n):
        # Collect same-season historical log-price changes
        seasonal_changes = []
        k = i - period
        while k >= 0:
            # Change from one year ago relative to the seasonal reference
            change = log_prices[k + period] - log_prices[k] if k + period <= i else 0.0
            # Expected seasonal log-price at this point
            seasonal_changes.append(log_prices[k])
            k -= period

        if len(seasonal_changes) < min_years:
            continue

        seasonal_changes = np.array(seasonal_changes)
        # Expected level: mean of historical same-season prices
        mu = np.mean(seasonal_changes)
        std = np.std(seasonal_changes)

        if std > 1e-9:
            sz[i] = np.clip((log_prices[i] - mu) / std, -4.0, 4.0)
            band_upper[i] = np.exp(mu + std)
            band_lower[i] = np.exp(mu - std)
        else:
            sz[i] = 0.0
            band_upper[i] = closes[i]
            band_lower[i] = closes[i]

    return sz, band_upper, band_lower
