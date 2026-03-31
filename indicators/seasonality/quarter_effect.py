"""Quarter-end effect — detects proximity to quarter-end dates and scores historical behavior."""
import numpy as np


def quarter_end_effect(closes: np.ndarray, datetimes: np.ndarray,
                       window: int = 10) -> tuple:
    """Quarter-end effect: detect proximity to quarter-end and historical behavior.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    datetimes : np.ndarray
        numpy datetime64 array.
    window : int
        Number of days before/after quarter-end to consider as "near quarter-end".

    Returns
    -------
    days_to_quarter_end : np.ndarray
        Approximate trading days until next quarter-end (Mar/Jun/Sep/Dec).
    quarter_end_score : np.ndarray
        Historical average return during the quarter-end window, z-scored.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    days_to_qe = np.full(n, np.nan)
    qe_score = np.full(n, np.nan)

    months = (datetimes.astype('datetime64[M]')
              - datetimes.astype('datetime64[Y]')).astype(int) + 1
    days = (datetimes.astype('datetime64[D]')
            - datetimes.astype('datetime64[M]')).astype(int) + 1

    # Quarter-end months: 3, 6, 9, 12
    qe_months = {3, 6, 9, 12}

    # Compute days to quarter end for each bar
    for i in range(n):
        m = months[i]
        # Next quarter-end month
        for qm in [3, 6, 9, 12]:
            if qm >= m:
                break
        else:
            qm = 3  # wrap to next year's March

        # Rough estimate: (qm - m) * 21 trading days per month + remaining days
        if qm >= m:
            months_away = qm - m
        else:
            months_away = (12 - m) + qm

        # Approximate trading days
        days_to_qe[i] = max(0, months_away * 21 - days[i] + 21)

    # Daily returns
    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    # Is near quarter end
    near_qe = days_to_qe <= window

    # Rolling score: average return during near-quarter-end periods
    for i in range(60, n):
        start = max(1, i - 252)
        mask = near_qe[start:i] & np.isfinite(rets[start:i])
        not_mask = (~near_qe[start:i]) & np.isfinite(rets[start:i])

        qe_rets = rets[start:i][mask]
        non_qe_rets = rets[start:i][not_mask]

        if len(qe_rets) < 5 or len(non_qe_rets) < 5:
            continue

        diff = np.mean(qe_rets) - np.mean(non_qe_rets)
        pooled_std = np.sqrt((np.var(qe_rets) + np.var(non_qe_rets)) / 2.0)
        if pooled_std > 0:
            qe_score[i] = np.clip(diff / pooled_std, -2.0, 2.0)
        else:
            qe_score[i] = 0.0

    return days_to_qe, qe_score
