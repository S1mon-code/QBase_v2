import numpy as np


def bayesian_online_trend(closes, hazard_rate=0.01):
    """Bayesian online changepoint detection (simplified Adams-MacKay).

    Maintains a run-length distribution and updates it bar-by-bar.
    The trend estimate is the slope of a simple linear fit within the
    current most-likely run.

    Parameters
    ----------
    closes : 1-D array of close prices.
    hazard_rate : prior probability of a changepoint at each bar (1/expected_run_length).

    Returns
    -------
    run_length : (N,) MAP estimate of current run length.
    changepoint_prob : (N,) probability that the current bar is a changepoint.
    trend_estimate : (N,) slope of linear fit over the current run.
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    run_length = np.full(n, np.nan, dtype=np.float64)
    changepoint_prob = np.full(n, np.nan, dtype=np.float64)
    trend_estimate = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return run_length, changepoint_prob, trend_estimate

    # Online sufficient statistics for Gaussian predictive distribution
    # We keep stats for each possible run length up to a max
    max_rl = min(n, 500)

    # Run-length probabilities: R[r] = P(run_length = r)
    R = np.zeros(max_rl + 1, dtype=np.float64)
    R[0] = 1.0  # start with run_length = 0

    # Sufficient stats per run length: mean, var (online Welford)
    mu = np.zeros(max_rl + 1, dtype=np.float64)
    var = np.ones(max_rl + 1, dtype=np.float64)  # prior variance
    count = np.zeros(max_rl + 1, dtype=np.float64)

    # Prior parameters
    mu0 = closes[0]
    var0 = 1.0  # will be updated after first few observations

    for t in range(n):
        x = closes[t]
        if np.isnan(x):
            continue

        # Update var0 adaptively using recent data
        if t > 1:
            recent = closes[max(0, t - 50): t + 1]
            recent = recent[~np.isnan(recent)]
            if len(recent) > 2:
                var0 = np.var(recent) + 1e-10

        # Predictive probability under each run length (Gaussian)
        pred_prob = np.zeros(max_rl + 1, dtype=np.float64)
        for r in range(min(t + 1, max_rl + 1)):
            if R[r] < 1e-20:
                continue
            if count[r] < 2:
                s2 = var0
            else:
                s2 = var[r] + var0 / (count[r] + 1)
            s2 = max(s2, 1e-10)
            diff = x - mu[r]
            pred_prob[r] = np.exp(-0.5 * diff * diff / s2) / np.sqrt(2 * np.pi * s2)

        # Growth probabilities
        growth = R[:max_rl + 1] * pred_prob * (1 - hazard_rate)

        # Changepoint probability
        cp_mass = np.sum(R[:max_rl + 1] * pred_prob * hazard_rate)

        # Shift run lengths: new R[r+1] = growth[r]
        new_R = np.zeros(max_rl + 1, dtype=np.float64)
        new_R[1: max_rl + 1] = growth[: max_rl]
        new_R[0] = cp_mass

        # Normalise
        total = new_R.sum()
        if total > 1e-20:
            new_R /= total

        # Update sufficient stats (shift)
        new_mu = np.zeros(max_rl + 1, dtype=np.float64)
        new_var = np.ones(max_rl + 1, dtype=np.float64) * var0
        new_count = np.zeros(max_rl + 1, dtype=np.float64)

        # r=0: new run starts
        new_mu[0] = x
        new_var[0] = var0
        new_count[0] = 1

        # r>0: continued runs
        limit = min(t + 1, max_rl)
        for r in range(limit):
            rn = r + 1
            if rn > max_rl:
                break
            new_count[rn] = count[r] + 1
            delta = x - mu[r]
            new_mu[rn] = mu[r] + delta / new_count[rn]
            if new_count[rn] > 1:
                new_var[rn] = var[r] + delta * (x - new_mu[rn])
            else:
                new_var[rn] = var0

        R = new_R
        mu = new_mu
        var = new_var
        count = new_count

        # MAP run length
        map_rl = int(np.argmax(R))
        run_length[t] = map_rl
        changepoint_prob[t] = R[0]

        # Trend estimate: slope over current run
        if map_rl >= 2:
            segment = closes[max(0, t - map_rl + 1): t + 1]
            segment = segment[~np.isnan(segment)]
            if len(segment) >= 2:
                xs = np.arange(len(segment), dtype=np.float64)
                xs -= xs.mean()
                trend_estimate[t] = np.dot(xs, segment) / (np.dot(xs, xs) + 1e-10)
            else:
                trend_estimate[t] = 0.0
        else:
            trend_estimate[t] = 0.0

    return run_length, changepoint_prob, trend_estimate
