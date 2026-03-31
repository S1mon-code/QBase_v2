import numpy as np


def robust_zscore(data, period=60):
    """Median Absolute Deviation (MAD) based z-score -- robust to outliers.

    Uses the rolling median and MAD instead of mean and std, making the
    z-score much more reliable for fat-tailed financial data.

    Parameters
    ----------
    data : 1-D array.
    period : rolling window size.

    Returns
    -------
    zscore : (N,) MAD-based z-score.
    mad : (N,) rolling Median Absolute Deviation.
    median : (N,) rolling median.
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    zscore = np.full(n, np.nan, dtype=np.float64)
    mad = np.full(n, np.nan, dtype=np.float64)
    median = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return zscore, mad, median

    # consistency constant for Gaussian equivalence
    MAD_SCALE = 1.4826

    for i in range(period - 1, n):
        win = data[i - period + 1: i + 1]
        if np.any(np.isnan(win)):
            valid = win[~np.isnan(win)]
            if len(valid) < 3:
                continue
        else:
            valid = win

        med = np.median(valid)
        m = np.median(np.abs(valid - med))

        median[i] = med
        mad[i] = m

        scaled_mad = m * MAD_SCALE
        if scaled_mad < 1e-10:
            zscore[i] = 0.0
        else:
            zscore[i] = (data[i] - med) / scaled_mad

    return zscore, mad, median
