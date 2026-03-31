"""Volatility seasonality — scores current realized volatility against historical same-month norms."""
import numpy as np


def vol_seasonality(closes: np.ndarray, datetimes: np.ndarray,
                    vol_period: int = 20) -> tuple:
    """Monthly volatility seasonal pattern.

    Computes rolling realized volatility, then for each month measures
    whether current vol is above or below the historical seasonal norm.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    datetimes : np.ndarray
        numpy datetime64 array.
    vol_period : int
        Window for realized volatility calculation.

    Returns
    -------
    vol_seasonal_score : np.ndarray
        Z-score of current vol vs historical same-month vol.
    expected_vol_rank : np.ndarray
        Percentile rank (0-1) of current month's expected vol among all months.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    vol_seasonal_score = np.full(n, np.nan)
    expected_vol_rank = np.full(n, np.nan)

    # Compute rolling realized vol
    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    rvol = np.full(n, np.nan)
    for i in range(vol_period, n):
        window = rets[i - vol_period + 1:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) >= vol_period // 2:
            rvol[i] = np.std(valid) * np.sqrt(252)

    # Month index
    months = (datetimes.astype('datetime64[M]')
              - datetimes.astype('datetime64[Y]')).astype(int) + 1

    # For each bar, compute stats of same-month vol from history
    min_lookback = 252 * 2
    for i in range(min_lookback, n):
        if not np.isfinite(rvol[i]):
            continue

        cur_month = months[i]
        # Collect same-month vols from history
        hist_mask = (months[:i] == cur_month) & np.isfinite(rvol[:i])
        same_month_vols = rvol[:i][hist_mask]

        if len(same_month_vols) < 10:
            continue

        mu = np.mean(same_month_vols)
        std = np.std(same_month_vols)
        if std > 1e-9:
            vol_seasonal_score[i] = np.clip((rvol[i] - mu) / std, -3.0, 3.0)
        else:
            vol_seasonal_score[i] = 0.0

        # Expected vol rank: where does this month's avg vol rank among all 12 months?
        all_month_avgs = []
        for m in range(1, 13):
            m_mask = (months[:i] == m) & np.isfinite(rvol[:i])
            m_vols = rvol[:i][m_mask]
            if len(m_vols) >= 5:
                all_month_avgs.append(np.mean(m_vols))
            else:
                all_month_avgs.append(np.nan)

        valid_avgs = [v for v in all_month_avgs if np.isfinite(v)]
        if len(valid_avgs) >= 6 and np.isfinite(all_month_avgs[cur_month - 1]):
            cur_avg = all_month_avgs[cur_month - 1]
            rank = np.sum(np.array(valid_avgs) <= cur_avg) / len(valid_avgs)
            expected_vol_rank[i] = rank

    return vol_seasonal_score, expected_vol_rank
