"""Weekday effect — computes day-of-week return patterns and z-scored historical weekday performance."""
import numpy as np


def weekday_effect(closes: np.ndarray, datetimes: np.ndarray,
                   lookback: int = 252) -> tuple:
    """Day-of-week return pattern.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    datetimes : np.ndarray
        numpy datetime64 array.
    lookback : int
        Rolling window of bars to compute weekday statistics.

    Returns
    -------
    weekday_score : np.ndarray
        Historical average return for the current day's weekday, z-scored.
    weekday_returns : np.ndarray
        Raw historical average return for the current weekday.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    weekday_score = np.full(n, np.nan)
    weekday_returns = np.full(n, np.nan)

    # Weekday: 0=Monday .. 6=Sunday
    # datetime64 epoch is Thursday (1970-01-01), so (days + 3) % 7 gives
    # 0=Mon..6=Sun
    days_since_epoch = (datetimes - np.datetime64('1970-01-01', 'D')).astype(int)
    weekdays = (days_since_epoch + 3) % 7

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    for i in range(2, n):
        start = max(1, i - lookback)
        cur_wd = weekdays[i]

        window_mask = (weekdays[start:i] == cur_wd) & np.isfinite(rets[start:i])
        matched = rets[start:i][window_mask]

        if len(matched) < 3:
            continue

        avg = np.mean(matched)
        weekday_returns[i] = avg

        std = np.std(matched)
        if std > 0.0:
            weekday_score[i] = np.clip(avg / std, -3.0, 3.0)
        else:
            weekday_score[i] = 0.0

    return weekday_score, weekday_returns
