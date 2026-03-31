import numpy as np


def target_encoded_regime(
    closes: np.ndarray,
    period: int = 60,
    n_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode price level into target-encoded bin by mean forward return.

    Divides trailing price range into ``n_bins`` bins and computes the mean
    forward return for each bin historically.  Returns the encoded value for
    the current bar's bin and the bin label.

    Parameters
    ----------
    closes : (N,) price series.
    period : lookback window.
    n_bins : number of price bins.

    Returns
    -------
    encoded_level : (N,) mean forward return of the current bar's bin.
    bin_label : (N,) bin index (0 to n_bins-1).
    """
    n = len(closes)
    encoded = np.full(n, np.nan, dtype=np.float64)
    bin_label = np.full(n, np.nan, dtype=np.float64)

    if n < period + 2:
        return encoded, bin_label

    # 1-bar forward return
    safe = np.maximum(closes, 1e-12)
    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    fwd_ret[:-1] = safe[1:] / safe[:-1] - 1.0

    for i in range(period + 1, n):
        window_c = closes[i - period : i]
        window_ret = fwd_ret[i - period : i]

        if np.any(np.isnan(window_c)) or np.any(np.isnan(window_ret)):
            continue

        lo = np.min(window_c)
        hi = np.max(window_c)
        if hi - lo < 1e-12:
            continue

        edges = np.linspace(lo, hi + 1e-12, n_bins + 1)
        # Assign each bar in the window to a bin
        bins = np.digitize(window_c, edges) - 1
        bins = np.clip(bins, 0, n_bins - 1)

        # Mean return per bin
        mean_per_bin = np.full(n_bins, 0.0)
        for b in range(n_bins):
            mask = bins == b
            if np.any(mask):
                mean_per_bin[b] = np.mean(window_ret[mask])

        # Current bar's bin
        cur_price = closes[i]
        cur_bin = int(np.clip(np.digitize(cur_price, edges) - 1, 0, n_bins - 1))
        bin_label[i] = float(cur_bin)
        encoded[i] = mean_per_bin[cur_bin]

    return encoded, bin_label
