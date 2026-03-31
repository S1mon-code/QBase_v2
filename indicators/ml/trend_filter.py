import numpy as np


def l1_trend_filter(closes, lambda_val=1.0):
    """L1 trend filtering (piecewise linear trend extraction).

    Approximated via Iteratively Reweighted Least Squares (IRLS).
    Solves:  min  0.5 * ||y - x||^2  +  lambda * ||D2 @ x||_1
    where D2 is the second-order difference operator.

    Parameters
    ----------
    closes : 1-D array of close prices.
    lambda_val : regularisation strength.  Higher = smoother trend with
        fewer breakpoints.

    Returns
    -------
    trend : (N,) extracted piecewise-linear trend.
    residual : (N,) closes - trend.
    breakpoints : (N,) binary indicator where the trend slope changes
        (1 = breakpoint, 0 = not).
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    trend = np.full(n, np.nan, dtype=np.float64)
    residual = np.full(n, np.nan, dtype=np.float64)
    breakpoints = np.full(n, np.nan, dtype=np.float64)

    if n < 4:
        return trend, residual, breakpoints

    # only process non-NaN prefix
    first_valid = 0
    while first_valid < n and np.isnan(closes[first_valid]):
        first_valid += 1
    if n - first_valid < 4:
        return trend, residual, breakpoints

    y = closes[first_valid:].copy()
    m = len(y)
    nan_mask = np.isnan(y)
    if nan_mask.any():
        # forward-fill NaNs for computation
        for i in range(1, m):
            if np.isnan(y[i]):
                y[i] = y[i - 1]

    # build second-order difference matrix D2 (m-2, m)
    # D2[i] = y[i] - 2*y[i+1] + y[i+2]
    # We work with the normal equation: (I + lambda * D2^T W D2) x = y
    # where W is diagonal reweighting for IRLS

    n_iter = 15
    eps_irls = 1e-4

    x = y.copy()  # initialise with raw data

    for iteration in range(n_iter):
        # compute second differences of current estimate
        d2x = x[:-2] - 2 * x[1:-1] + x[2:]

        # IRLS weights: w_i = 1 / max(|d2x_i|, eps)
        w = 1.0 / np.maximum(np.abs(d2x), eps_irls)

        # solve (I + lambda * D2^T diag(w) D2) x = y
        # build tridiagonal-ish system
        # D2^T diag(w) D2 is banded with bandwidth 2

        # construct the matrix as dense (manageable for typical N)
        # for large N, use banded solver
        if m > 5000:
            # use simplified approach for very large arrays
            # just do a few Jacobi-like iterations
            rhs = y.copy()
            for _ in range(50):
                x_new = rhs.copy()
                for i in range(2, m):
                    x_new[i - 2] += lambda_val * w[i - 2] * (x[i - 2] - 2 * x[i - 1] + x[i])
                # This is approximate; for production use scipy banded solver
                # Simplified: weighted average
                alpha = 0.5
                x = (1 - alpha) * x + alpha * x_new
            break

        # dense approach
        A = np.eye(m, dtype=np.float64)

        for i in range(m - 2):
            wi = lambda_val * w[i]
            A[i, i] += wi
            A[i, i + 1] -= 2 * wi
            A[i, i + 2] += wi
            A[i + 1, i] -= 2 * wi
            A[i + 1, i + 1] += 4 * wi
            A[i + 1, i + 2] -= 2 * wi
            A[i + 2, i] += wi
            A[i + 2, i + 1] -= 2 * wi
            A[i + 2, i + 2] += wi

        try:
            x = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            x = y.copy()
            break

    trend[first_valid:] = x
    residual[first_valid:] = closes[first_valid:] - x

    # detect breakpoints: where second difference of trend exceeds threshold
    bp = np.zeros(m, dtype=np.float64)
    if m >= 3:
        d2_trend = np.abs(x[:-2] - 2 * x[1:-1] + x[2:])
        threshold = np.median(d2_trend) + 2 * np.std(d2_trend)
        if threshold < 1e-10:
            threshold = 1e-10
        bp[1:-1] = (d2_trend > threshold).astype(np.float64)
    breakpoints[first_valid:] = bp

    return trend, residual, breakpoints
