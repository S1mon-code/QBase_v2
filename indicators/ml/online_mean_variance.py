import numpy as np


def welford_stats(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welford's online algorithm for running mean/variance.

    Numerically stable single-pass computation.  Each output[i] reflects
    the mean/variance of data[0:i+1].

    Parameters
    ----------
    data : (N,) input series.

    Returns
    -------
    running_mean : (N,) cumulative mean up to bar i.
    running_var : (N,) cumulative variance (population) up to bar i.
    running_std : (N,) cumulative standard deviation up to bar i.
    """
    n = len(data)
    running_mean = np.full(n, np.nan, dtype=np.float64)
    running_var = np.full(n, np.nan, dtype=np.float64)
    running_std = np.full(n, np.nan, dtype=np.float64)

    mean = 0.0
    m2 = 0.0
    count = 0

    for i in range(n):
        x = data[i]
        if np.isnan(x):
            if count > 0:
                running_mean[i] = mean
                running_var[i] = m2 / count if count > 0 else 0.0
                running_std[i] = np.sqrt(running_var[i])
            continue

        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        m2 += delta * delta2

        running_mean[i] = mean
        running_var[i] = m2 / count if count > 0 else 0.0
        running_std[i] = np.sqrt(running_var[i])

    return running_mean, running_var, running_std
