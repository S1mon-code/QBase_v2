"""Turn-of-month effect — measures return concentration around month-end and month-start boundaries."""
import numpy as np


def month_turn(closes: np.ndarray, datetimes: np.ndarray,
               window: int = 3) -> tuple:
    """Turn-of-month effect: last ``window`` + first ``window`` trading days.

    The turn-of-month effect is a well-documented anomaly where returns
    concentrate around month boundaries.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    datetimes : np.ndarray
        numpy datetime64 array.
    window : int
        Number of days before and after month turn to consider.

    Returns
    -------
    tom_score : np.ndarray
        Rolling average turn-of-month return z-scored.
    is_month_turn : np.ndarray (bool)
        True if current bar falls within the turn-of-month window.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    tom_score = np.full(n, np.nan)
    is_tom = np.zeros(n, dtype=bool)

    months = datetimes.astype('datetime64[M]')
    days = (datetimes - months).astype(int) + 1  # day of month (1-based)

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    # Mark turn-of-month bars
    for i in range(1, n):
        dom = days[i]
        # First ``window`` days of month
        if dom <= window:
            is_tom[i] = True
            continue
        # Last ``window`` days: check if month changes within next few bars
        if i + 1 < n and months[i + 1] != months[i]:
            is_tom[i] = True
            continue
        # Also check if current bar is within ``window`` bars of month end
        # by looking backwards for month change
        if i >= 1 and months[i] != months[i - 1]:
            is_tom[i] = True
            continue

    # For last days detection: mark ``window`` bars before month change
    for i in range(n - 1, 0, -1):
        if i + 1 < n and months[i + 1] != months[i]:
            # This is the last day of a month
            for k in range(max(0, i - window + 1), i + 1):
                is_tom[k] = True

    # Rolling tom_score: average return on TOM days vs non-TOM days
    lookback = 252
    for i in range(2, n):
        start = max(1, i - lookback)
        tom_mask = is_tom[start:i] & np.isfinite(rets[start:i])
        non_tom_mask = (~is_tom[start:i]) & np.isfinite(rets[start:i])

        tom_vals = rets[start:i][tom_mask]
        non_vals = rets[start:i][non_tom_mask]

        if len(tom_vals) < 5 or len(non_vals) < 5:
            continue

        diff = np.mean(tom_vals) - np.mean(non_vals)
        pooled_std = np.sqrt(
            (np.var(tom_vals) * len(tom_vals) + np.var(non_vals) * len(non_vals))
            / (len(tom_vals) + len(non_vals))
        )
        if pooled_std > 0:
            tom_score[i] = np.clip(diff / pooled_std, -3.0, 3.0)
        else:
            tom_score[i] = 0.0

    return tom_score, is_tom
