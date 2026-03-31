import numpy as np


def momentum_components(closes, short=5, medium=20, long=60):
    """Decompose momentum into short/medium/long-term components.

    Each component is the rate of change over its respective period,
    normalised by rolling volatility for comparability.  The composite
    score weights the three components by their recent predictive power
    (rolling correlation with next-bar returns).

    Parameters
    ----------
    closes : 1-D array of close prices.
    short : short momentum period.
    medium : medium momentum period.
    long : long momentum period.

    Returns
    -------
    short_mom : (N,) short-term momentum (vol-adjusted ROC).
    medium_mom : (N,) medium-term momentum.
    long_mom : (N,) long-term momentum.
    composite_score : (N,) weighted combination using adaptive weights.
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    short_mom = np.full(n, np.nan, dtype=np.float64)
    medium_mom = np.full(n, np.nan, dtype=np.float64)
    long_mom = np.full(n, np.nan, dtype=np.float64)
    composite_score = np.full(n, np.nan, dtype=np.float64)

    if n < long + 1:
        return short_mom, medium_mom, long_mom, composite_score

    # compute log returns
    log_ret = np.full(n, np.nan, dtype=np.float64)
    log_ret[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-10))

    # rolling volatility (for normalisation), use medium period
    vol = np.full(n, np.nan, dtype=np.float64)
    for i in range(medium - 1, n):
        win = log_ret[i - medium + 1: i + 1]
        if np.any(np.isnan(win)):
            continue
        vol[i] = win.std()

    # momentum components (ROC normalised by vol)
    for i in range(long, n):
        v = vol[i]
        if np.isnan(v) or v < 1e-10:
            continue

        if closes[i - short] > 0:
            short_mom[i] = (closes[i] / closes[i - short] - 1) / v
        if closes[i - medium] > 0:
            medium_mom[i] = (closes[i] / closes[i - medium] - 1) / v
        if closes[i - long] > 0:
            long_mom[i] = (closes[i] / closes[i - long] - 1) / v

    # adaptive weighting: rolling correlation of each component with
    # next-bar return (using past data only -- no look-ahead)
    eval_period = medium  # lookback for evaluating predictive power
    for i in range(long + eval_period, n):
        s = short_mom[i - eval_period: i]
        m = medium_mom[i - eval_period: i]
        lg = long_mom[i - eval_period: i]
        fwd = log_ret[i - eval_period + 1: i + 1]  # returns that follow each signal

        if np.any(np.isnan(s)) or np.any(np.isnan(m)) or np.any(np.isnan(lg)) or np.any(np.isnan(fwd)):
            # fallback: equal weights
            if not (np.isnan(short_mom[i]) or np.isnan(medium_mom[i]) or np.isnan(long_mom[i])):
                composite_score[i] = (short_mom[i] + medium_mom[i] + long_mom[i]) / 3.0
            continue

        def _corr(a, b):
            a_m = a - a.mean()
            b_m = b - b.mean()
            denom = np.sqrt(np.dot(a_m, a_m) * np.dot(b_m, b_m))
            if denom < 1e-10:
                return 0.0
            return np.dot(a_m, b_m) / denom

        w_s = abs(_corr(s, fwd))
        w_m = abs(_corr(m, fwd))
        w_l = abs(_corr(lg, fwd))
        w_total = w_s + w_m + w_l

        if w_total < 1e-10:
            w_s = w_m = w_l = 1.0 / 3.0
        else:
            w_s /= w_total
            w_m /= w_total
            w_l /= w_total

        if not (np.isnan(short_mom[i]) or np.isnan(medium_mom[i]) or np.isnan(long_mom[i])):
            composite_score[i] = w_s * short_mom[i] + w_m * medium_mom[i] + w_l * long_mom[i]

    return short_mom, medium_mom, long_mom, composite_score
