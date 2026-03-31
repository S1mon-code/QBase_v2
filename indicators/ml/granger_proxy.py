import numpy as np


def granger_causality_score(series_a, series_b, period=60, max_lag=5):
    """Rolling Granger causality F-statistic proxy.

    For each rolling window, fits two OLS regressions for each direction
    (A->B and B->A) and returns the larger F-statistic together with the
    inferred direction.

    Parameters
    ----------
    series_a, series_b : 1-D arrays of equal length.
    period : rolling window size.
    max_lag : maximum number of lags to include.

    Returns
    -------
    f_score : (N,) F-statistic of the dominant direction.
    direction : (N,) 1 = A causes B, -1 = B causes A, 0 = neither.
    """
    series_a = np.asarray(series_a, dtype=np.float64)
    series_b = np.asarray(series_b, dtype=np.float64)
    n = len(series_a)
    f_score = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, np.nan, dtype=np.float64)

    if n < period + max_lag:
        return f_score, direction

    def _ols_f(y, X_restricted, X_full):
        """F-test comparing restricted vs full model."""
        # restricted: y ~ X_restricted
        # full:       y ~ X_full
        n_obs = len(y)
        k_r = X_restricted.shape[1]
        k_f = X_full.shape[1]

        # solve via normal equations
        try:
            beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
            beta_f = np.linalg.lstsq(X_full, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0

        rss_r = np.sum((y - X_restricted @ beta_r) ** 2)
        rss_f = np.sum((y - X_full @ beta_f) ** 2)

        df_num = k_f - k_r
        df_den = n_obs - k_f
        if df_den <= 0 or df_num <= 0 or rss_f < 1e-15:
            return 0.0
        f = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
        return max(0.0, f)

    def _granger_one_dir(cause, effect, lags):
        """F-stat for cause -> effect."""
        m = len(cause) - lags
        if m < lags + 5:
            return 0.0
        y = effect[lags:]
        # restricted: own lags of effect
        X_r = np.column_stack([effect[lags - j - 1: -j - 1] for j in range(lags)])
        # full: own lags + cause lags
        X_f = np.column_stack([X_r] + [cause[lags - j - 1: -j - 1] for j in range(lags)])
        return _ols_f(y, X_r, X_f)

    for i in range(period + max_lag - 1, n):
        a_win = series_a[i - period - max_lag + 1: i + 1]
        b_win = series_b[i - period - max_lag + 1: i + 1]
        if np.any(np.isnan(a_win)) or np.any(np.isnan(b_win)):
            continue

        f_ab = _granger_one_dir(a_win, b_win, max_lag)  # A -> B
        f_ba = _granger_one_dir(b_win, a_win, max_lag)  # B -> A

        f_threshold = 2.0  # rough significance proxy
        if f_ab > f_ba and f_ab > f_threshold:
            f_score[i] = f_ab
            direction[i] = 1.0
        elif f_ba > f_ab and f_ba > f_threshold:
            f_score[i] = f_ba
            direction[i] = -1.0
        else:
            f_score[i] = max(f_ab, f_ba)
            direction[i] = 0.0

    return f_score, direction
