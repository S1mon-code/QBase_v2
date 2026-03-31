import numpy as np


def conditional_volatility(
    closes: np.ndarray,
    period: int = 20,
    threshold: float = 0.0,
) -> tuple:
    """Separate upside vs downside volatility.

    Computes rolling standard deviation of returns above threshold (up_vol)
    and below threshold (down_vol) independently.
    vol_asymmetry = down_vol - up_vol. Positive means more downside risk.
    Returns (up_vol, down_vol, vol_asymmetry).
    """
    n = len(closes)
    if n < 2:
        emp = np.full(n, np.nan)
        return emp.copy(), emp.copy(), emp.copy()

    closes = closes.astype(np.float64)
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(closes[1:] / closes[:-1])

    up_vol = np.full(n, np.nan)
    down_vol = np.full(n, np.nan)
    vol_asymmetry = np.full(n, np.nan)

    for i in range(period, n):
        window = log_ret[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue

        up_rets = valid[valid > threshold]
        down_rets = valid[valid < threshold]

        if len(up_rets) >= 2:
            up_vol[i] = np.std(up_rets, ddof=1)
        else:
            up_vol[i] = 0.0

        if len(down_rets) >= 2:
            down_vol[i] = np.std(down_rets, ddof=1)
        else:
            down_vol[i] = 0.0

        vol_asymmetry[i] = down_vol[i] - up_vol[i]

    return up_vol, down_vol, vol_asymmetry
