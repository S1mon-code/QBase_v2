"""Intraweek momentum pattern — compares early-week (Mon-Wed) vs late-week (Thu-Fri) return tendencies."""
import numpy as np


def intraweek_momentum(closes: np.ndarray, datetimes: np.ndarray,
                       lookback: int = 52) -> tuple:
    """Mon-to-Wed vs Thu-Fri momentum pattern.

    Computes rolling averages of early-week (Mon-Wed) and late-week (Thu-Fri)
    returns over ``lookback`` weeks and returns z-scored scores for each.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    datetimes : np.ndarray
        numpy datetime64 array.
    lookback : int
        Number of weeks of history to average over.

    Returns
    -------
    early_week_score : np.ndarray
        Z-scored average Mon-Wed return.
    late_week_score : np.ndarray
        Z-scored average Thu-Fri return.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    early_score = np.full(n, np.nan)
    late_score = np.full(n, np.nan)

    days_since_epoch = (datetimes - np.datetime64('1970-01-01', 'D')).astype(int)
    weekdays = (days_since_epoch + 3) % 7  # 0=Mon .. 6=Sun

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    # Approximate bars per week (5 trading days)
    window = lookback * 5

    for i in range(2, n):
        start = max(1, i - window)
        wd = weekdays[i]

        # Early week: Mon(0), Tue(1), Wed(2)
        # Late week: Thu(3), Fri(4)
        early_mask = np.isin(weekdays[start:i], [0, 1, 2]) & np.isfinite(rets[start:i])
        late_mask = np.isin(weekdays[start:i], [3, 4]) & np.isfinite(rets[start:i])

        early_vals = rets[start:i][early_mask]
        late_vals = rets[start:i][late_mask]

        if len(early_vals) >= 5:
            mu_e = np.mean(early_vals)
            std_e = np.std(early_vals)
            if std_e > 0:
                early_score[i] = np.clip(mu_e / std_e, -3.0, 3.0)
            else:
                early_score[i] = 0.0

        if len(late_vals) >= 5:
            mu_l = np.mean(late_vals)
            std_l = np.std(late_vals)
            if std_l > 0:
                late_score[i] = np.clip(mu_l / std_l, -3.0, 3.0)
            else:
                late_score[i] = 0.0

    return early_score, late_score
