import numpy as np


def garch_like_forecast(closes, period=60, alpha=0.1, beta=0.85):
    """Simple GARCH(1,1)-like volatility forecast (no arch package needed).

    sigma^2(t) = omega + alpha * r^2(t-1) + beta * sigma^2(t-1)

    omega is calibrated so that the unconditional variance matches the
    rolling sample variance: omega = var * (1 - alpha - beta).

    Parameters
    ----------
    closes : 1-D array of close prices.
    period : lookback for calibrating omega (sample variance).
    alpha : ARCH coefficient (reaction to shocks).
    beta : GARCH coefficient (persistence).

    Returns
    -------
    vol_forecast : (N,) 1-step ahead volatility forecast (annualised, 252-day).
    vol_ratio : (N,) ratio of GARCH forecast to rolling realised volatility.
        >1 = GARCH expects higher vol than recent realised.
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    vol_forecast = np.full(n, np.nan, dtype=np.float64)
    vol_ratio = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return vol_forecast, vol_ratio

    # log returns
    log_ret = np.full(n, np.nan, dtype=np.float64)
    log_ret[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-10))

    # ensure alpha + beta < 1
    if alpha + beta >= 1.0:
        beta = 0.99 - alpha

    sigma2 = 0.0
    ann = np.sqrt(252)

    for i in range(period, n):
        r = log_ret[i]
        if np.isnan(r):
            continue

        # rolling sample variance for omega calibration
        win = log_ret[i - period + 1: i + 1]
        valid = win[~np.isnan(win)]
        if len(valid) < 5:
            continue
        sample_var = np.var(valid)

        omega = sample_var * (1 - alpha - beta)
        omega = max(omega, 1e-12)

        if i == period:
            sigma2 = sample_var

        sigma2 = omega + alpha * r * r + beta * sigma2
        sigma2 = max(sigma2, 1e-12)

        vol_forecast[i] = np.sqrt(sigma2) * ann

        # realised vol
        real_vol = np.std(valid) * ann
        if real_vol > 1e-10:
            vol_ratio[i] = (np.sqrt(sigma2) * ann) / real_vol
        else:
            vol_ratio[i] = 1.0

    return vol_forecast, vol_ratio
