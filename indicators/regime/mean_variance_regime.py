import numpy as np


def mv_regime(
    data: np.ndarray,
    period: int = 60,
    n_regimes: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple mean-variance regime detection using rolling windows.

    Classifies each bar into a volatility regime by comparing
    the current rolling volatility to historical percentiles.

    Returns (regime_label, regime_mean, regime_vol).
    Labels: 0=low_vol, 1=normal, 2=high_vol (for n_regimes=3).
    """
    n = len(data)
    regime_label = np.full(n, np.nan, dtype=np.float64)
    regime_mean = np.full(n, np.nan, dtype=np.float64)
    regime_vol = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return regime_label, regime_mean, regime_vol

    safe = np.maximum(data, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    if len(log_ret) < period:
        return regime_label, regime_mean, regime_vol

    # Precompute rolling stats
    rolling_means = np.full(len(log_ret), np.nan, dtype=np.float64)
    rolling_vols = np.full(len(log_ret), np.nan, dtype=np.float64)

    for i in range(period - 1, len(log_ret)):
        window = log_ret[i - period + 1 : i + 1]
        rolling_means[i] = np.mean(window)
        rolling_vols[i] = np.std(window, ddof=1)

    # Need enough history to compute percentiles
    lookback = period * 3
    if len(log_ret) < lookback:
        lookback = len(log_ret)

    # Compute percentile boundaries for regime classification
    boundaries = np.linspace(0, 100, n_regimes + 1)[1:-1]  # e.g., [33.3, 66.7]

    for i in range(period - 1, len(log_ret)):
        cur_vol = rolling_vols[i]
        cur_mean = rolling_means[i]

        if np.isnan(cur_vol):
            continue

        # Historical vol distribution for percentile comparison
        hist_start = max(period - 1, i - lookback)
        hist_vols = rolling_vols[hist_start : i + 1]
        valid_vols = hist_vols[~np.isnan(hist_vols)]

        if len(valid_vols) < 5:
            continue

        # Determine regime based on percentile rank
        pct_rank = np.sum(valid_vols <= cur_vol) / len(valid_vols) * 100.0

        label = 0
        for b_idx, b in enumerate(boundaries):
            if pct_rank > b:
                label = b_idx + 1

        # Map back: log_ret index i -> data index i+1
        idx = i + 1
        if idx < n:
            regime_label[idx] = float(label)
            regime_mean[idx] = cur_mean
            regime_vol[idx] = cur_vol

    return regime_label, regime_mean, regime_vol
