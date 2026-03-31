"""Cross-asset return dispersion indicator.

Measures how differently assets in a basket are behaving.  High
dispersion implies low correlation and good diversification
opportunities.
"""

import numpy as np


def dispersion(
    closes_list: list[np.ndarray],
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-asset return dispersion.

    Parameters
    ----------
    closes_list : list of close-price arrays (one per asset).
                  All arrays must have the same length.
    period      : lookback for dispersion and correlation estimation.

    Returns
    -------
    (dispersion_val, avg_corr, dispersion_zscore)
        dispersion_val    – cross-sectional standard deviation of
                            rolling returns across assets.
        avg_corr          – average pairwise rolling correlation.
        dispersion_zscore – rolling z-score of dispersion_val.
    """
    if not closes_list:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    n = len(closes_list[0])
    k = len(closes_list)
    if n == 0 or k < 2:
        empty = np.full(n, np.nan) if n > 0 else np.array([], dtype=float)
        return empty, empty.copy(), empty.copy()

    # Compute returns for each asset
    rets = np.full((k, n), np.nan)
    for j in range(k):
        c = closes_list[j]
        safe_c = np.where(c == 0, np.nan, c)
        rets[j, 1:] = safe_c[1:] / safe_c[:-1] - 1.0

    disp = np.full(n, np.nan)
    avg_corr = np.full(n, np.nan)
    disp_z = np.full(n, np.nan)

    for i in range(period, n):
        # Cross-sectional dispersion of period-returns
        period_rets = np.full(k, np.nan)
        for j in range(k):
            window = rets[j, i - period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                period_rets[j] = np.sum(valid)

        valid_pr = period_rets[~np.isnan(period_rets)]
        if len(valid_pr) >= 2:
            disp[i] = np.std(valid_pr, ddof=1)

        # Average pairwise correlation over the window
        corr_sum = 0.0
        corr_count = 0
        for a in range(k):
            for b in range(a + 1, k):
                ra = rets[a, i - period + 1 : i + 1]
                rb = rets[b, i - period + 1 : i + 1]
                mask = ~(np.isnan(ra) | np.isnan(rb))
                if np.sum(mask) < 5:
                    continue
                va, vb = ra[mask], rb[mask]
                sa, sb = np.std(va, ddof=1), np.std(vb, ddof=1)
                if sa > 0 and sb > 0:
                    corr_sum += np.corrcoef(va, vb)[0, 1]
                    corr_count += 1

        if corr_count > 0:
            avg_corr[i] = corr_sum / corr_count

    # Z-score of dispersion
    disp_period = max(period, 60)
    for i in range(disp_period - 1, n):
        window = disp[i - disp_period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            disp_z[i] = (disp[i] - mu) / sigma

    return disp, avg_corr, disp_z
