import numpy as np


def squeeze_probability(closes: np.ndarray, oi: np.ndarray,
                        volumes: np.ndarray, period: int = 20) -> tuple:
    """Short/long squeeze probability estimator.

    Detects conditions conducive to squeezes:
    - Short squeeze: high OI + price rising + volume surge + OI declining
    - Long squeeze: high OI + price falling + volume surge + OI declining

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volumes.
    period : int
        Lookback period.

    Returns
    -------
    short_squeeze_prob : np.ndarray
        Short squeeze probability (0-1).
    long_squeeze_prob : np.ndarray
        Long squeeze probability (0-1).
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    ss_prob = np.full(n, np.nan)
    ls_prob = np.full(n, np.nan)

    if n <= period:
        return ss_prob, ls_prob

    # Price momentum
    price_mom = np.full(n, np.nan)
    price_mom[period:] = closes[period:] / closes[:-period] - 1.0

    # OI change rate
    oi_chg = np.full(n, np.nan)
    oi_chg[period:] = (oi[period:] - oi[:-period]) / np.maximum(oi[:-period], 1.0)

    # Volume surge
    vol_avg = np.full(n, np.nan)
    for i in range(period, n):
        window = volumes[i - period:i]
        valid = window[np.isfinite(window)]
        if len(valid) > 0:
            vol_avg[i] = np.mean(valid)

    for i in range(period * 2, n):
        pm = price_mom[i]
        oc = oi_chg[i]
        va = vol_avg[i]

        if np.isnan(pm) or np.isnan(oc) or np.isnan(va) or va == 0:
            continue

        vol_ratio = volumes[i] / va if np.isfinite(volumes[i]) else 1.0

        # OI percentile (high OI = more fuel for squeeze)
        oi_window = oi[i - period * 2:i + 1]
        valid_oi = oi_window[np.isfinite(oi_window)]
        if len(valid_oi) < 10:
            continue
        oi_pctl = np.sum(valid_oi <= oi[i]) / len(valid_oi)

        # OI declining = positions being forced out
        oi_declining = max(0.0, -oc)

        # Volume surge = panic
        vol_surge = min(1.0, max(0.0, (vol_ratio - 1.0) / 2.0))

        # Short squeeze: price rising + high OI + OI declining + volume surge
        if pm > 0:
            price_score = min(1.0, pm / 0.10)  # 10% move = max
            ss_prob[i] = np.clip(
                price_score * 0.3 + oi_pctl * 0.2 + oi_declining * 0.25 + vol_surge * 0.25,
                0.0, 1.0
            )
        else:
            ss_prob[i] = 0.0

        # Long squeeze: price falling + high OI + OI declining + volume surge
        if pm < 0:
            price_score = min(1.0, abs(pm) / 0.10)
            ls_prob[i] = np.clip(
                price_score * 0.3 + oi_pctl * 0.2 + oi_declining * 0.25 + vol_surge * 0.25,
                0.0, 1.0
            )
        else:
            ls_prob[i] = 0.0

    return ss_prob, ls_prob
