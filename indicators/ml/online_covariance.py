import numpy as np


def exponential_covariance(returns_matrix, halflife=20):
    """Exponentially weighted rolling covariance matrix.

    Computes EWMA covariance bar-by-bar using the decay factor derived from
    ``halflife``.

    Parameters
    ----------
    returns_matrix : (N, K) array of return series.
    halflife : EWMA halflife in bars.

    Returns
    -------
    variances : (N, K) EWMA variance for each series.
    correlations : (N,) average pairwise EWMA correlation at each bar.
    """
    returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
    n, k = returns_matrix.shape
    variances = np.full((n, k), np.nan, dtype=np.float64)
    correlations = np.full(n, np.nan, dtype=np.float64)

    if n < 2 or k < 1:
        return variances, correlations

    alpha = 1 - np.exp(-np.log(2) / halflife)

    # initialise with first non-NaN observation
    ewma_mean = np.zeros(k, dtype=np.float64)
    ewma_cov = np.zeros((k, k), dtype=np.float64)
    initialised = False

    for i in range(n):
        row = returns_matrix[i]
        if np.any(np.isnan(row)):
            continue

        if not initialised:
            ewma_mean = row.copy()
            initialised = True
            variances[i] = 0.0
            if k > 1:
                correlations[i] = 0.0
            continue

        # update mean
        delta = row - ewma_mean
        ewma_mean = ewma_mean + alpha * delta

        # update covariance
        ewma_cov = (1 - alpha) * (ewma_cov + alpha * np.outer(delta, delta))

        # extract variances
        diag = np.diag(ewma_cov)
        variances[i] = diag

        # average pairwise correlation
        if k > 1:
            stds = np.sqrt(np.maximum(diag, 1e-20))
            corr_sum = 0.0
            count = 0
            for a in range(k):
                for b in range(a + 1, k):
                    c = ewma_cov[a, b] / (stds[a] * stds[b])
                    corr_sum += c
                    count += 1
            correlations[i] = corr_sum / count if count > 0 else 0.0
        else:
            correlations[i] = 1.0

    return variances, correlations
