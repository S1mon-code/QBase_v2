"""Expiry week effect — detects futures expiry weeks and scores historical return patterns."""
import numpy as np


def expiry_week_effect(closes: np.ndarray, datetimes: np.ndarray,
                       lookback: int = 52) -> tuple:
    """Futures expiry week behavior pattern.

    Chinese futures typically expire around the 15th of the delivery month.
    This detects the 3rd week of each month as a proxy for expiry week and
    measures historical return patterns during that period.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    datetimes : np.ndarray
        numpy datetime64 array.
    lookback : int
        Number of weeks of history for rolling average.

    Returns
    -------
    expiry_score : np.ndarray
        Z-scored average return during expiry weeks.
    is_expiry_week : np.ndarray (bool)
        True if current bar is in an estimated expiry week (days 11-17).
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    expiry_score = np.full(n, np.nan)
    is_expiry = np.zeros(n, dtype=bool)

    months = datetimes.astype('datetime64[M]')
    days = (datetimes - months).astype(int) + 1  # day of month

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    # Mark expiry week: days 11-17 of each month (around 3rd week)
    for i in range(n):
        if 11 <= days[i] <= 17:
            is_expiry[i] = True

    # Rolling score
    window = lookback * 5  # approx bars
    for i in range(2, n):
        start = max(1, i - window)

        exp_mask = is_expiry[start:i] & np.isfinite(rets[start:i])
        non_exp_mask = (~is_expiry[start:i]) & np.isfinite(rets[start:i])

        exp_vals = rets[start:i][exp_mask]
        non_vals = rets[start:i][non_exp_mask]

        if len(exp_vals) < 5 or len(non_vals) < 5:
            continue

        diff = np.mean(exp_vals) - np.mean(non_vals)
        pooled_std = np.sqrt(
            (np.var(exp_vals) * len(exp_vals) + np.var(non_vals) * len(non_vals))
            / (len(exp_vals) + len(non_vals))
        )
        if pooled_std > 0:
            expiry_score[i] = np.clip(diff / pooled_std, -3.0, 3.0)
        else:
            expiry_score[i] = 0.0

    return expiry_score, is_expiry
