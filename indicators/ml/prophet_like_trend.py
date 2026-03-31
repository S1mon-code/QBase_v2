import numpy as np


def piecewise_trend(
    closes: np.ndarray,
    n_changepoints: int = 5,
    period: int = 252,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simplified Prophet-like piecewise linear trend detection.

    Fits a piecewise linear model with ``n_changepoints`` on a trailing
    window of ``period`` bars.  Changepoints are placed at uniform quantiles
    in the window and their magnitudes are estimated via least squares.
    Retrains every ``period // 4`` bars.

    Parameters
    ----------
    closes : (N,) price series.
    n_changepoints : number of potential changepoints.
    period : lookback window length.

    Returns
    -------
    trend : (N,) fitted piecewise linear trend value.
    growth_rate : (N,) instantaneous growth rate (slope at current bar).
    changepoint_indicator : (N,) 1.0 if nearest changepoint had a significant
        adjustment, 0.0 otherwise.
    """
    n = len(closes)
    trend = np.full(n, np.nan, dtype=np.float64)
    growth_rate = np.full(n, np.nan, dtype=np.float64)
    changepoint_ind = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return trend, growth_rate, changepoint_ind

    retrain_every = max(1, period // 4)
    coefs: np.ndarray | None = None
    cp_positions: np.ndarray | None = None
    win_start = 0

    for i in range(period, n):
        need_train = coefs is None or (i - period) % retrain_every == 0

        if need_train:
            window = closes[i - period : i].astype(np.float64)
            if np.any(np.isnan(window)):
                continue

            win_start = i - period
            # Place changepoints at uniform quantiles
            cp_positions = np.linspace(0, period - 1, n_changepoints + 2)[1:-1].astype(int)

            # Build design matrix: [1, t, max(0, t-cp1), max(0, t-cp2), ...]
            t = np.arange(period, dtype=np.float64)
            X = np.ones((period, 2 + len(cp_positions)), dtype=np.float64)
            X[:, 1] = t
            for j, cp in enumerate(cp_positions):
                X[:, 2 + j] = np.maximum(0.0, t - cp)

            # Least squares
            try:
                coefs, _, _, _ = np.linalg.lstsq(X, window, rcond=None)
            except np.linalg.LinAlgError:
                continue

        if coefs is not None and cp_positions is not None:
            # Evaluate at bar i
            t_cur = float(i - win_start)
            x_cur = np.ones(2 + len(cp_positions), dtype=np.float64)
            x_cur[1] = t_cur
            for j, cp in enumerate(cp_positions):
                x_cur[2 + j] = max(0.0, t_cur - cp)

            trend[i] = x_cur @ coefs

            # Growth rate = derivative = base slope + active changepoint deltas
            slope = coefs[1]
            for j, cp in enumerate(cp_positions):
                if t_cur > cp:
                    slope += coefs[2 + j]
            growth_rate[i] = slope

            # Changepoint indicator: find nearest changepoint and check magnitude
            if len(cp_positions) > 0:
                nearest_idx = np.argmin(np.abs(cp_positions - (t_cur % period)))
                cp_mag = abs(coefs[2 + nearest_idx])
                threshold = 0.1 * abs(coefs[1]) if abs(coefs[1]) > 1e-12 else 1e-6
                changepoint_ind[i] = 1.0 if cp_mag > threshold else 0.0
            else:
                changepoint_ind[i] = 0.0

    return trend, growth_rate, changepoint_ind
