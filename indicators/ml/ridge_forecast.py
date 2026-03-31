import numpy as np
from sklearn.linear_model import Ridge


def rolling_ridge(
    closes: np.ndarray,
    features_matrix: np.ndarray,
    period: int = 120,
    forecast_horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Ridge regression: predict next-bar return from features.

    Trains on a trailing window of ``period`` bars using forward returns as
    the target.  Retrains every ``period // 4`` bars.

    Parameters
    ----------
    closes : (N,) price series.
    features_matrix : (N, K) feature array.
    period : training window length.
    forecast_horizon : number of bars ahead for the target return.

    Returns
    -------
    predictions : (N,) predicted return for the next bar.
    r_squared : (N,) in-sample R² of the last trained model.
    """
    n = len(closes)
    k = features_matrix.shape[1]
    predictions = np.full(n, np.nan, dtype=np.float64)
    r_squared = np.full(n, np.nan, dtype=np.float64)

    if n < period + forecast_horizon:
        return predictions, r_squared

    # Precompute forward returns (log)
    safe = np.maximum(closes, 1e-12)
    log_p = np.log(safe)
    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    fwd_ret[: n - forecast_horizon] = log_p[forecast_horizon:] - log_p[: n - forecast_horizon]

    retrain_every = max(1, period // 4)
    model: Ridge | None = None
    mean_x: np.ndarray | None = None
    std_x: np.ndarray | None = None

    for i in range(period + forecast_horizon, n):
        need_train = model is None or (i - period - forecast_horizon) % retrain_every == 0

        if need_train:
            # Training window: features and corresponding forward returns
            # Use bars [i - period - forecast_horizon, i - forecast_horizon)
            # so that the target returns are realised and not future-leaking.
            start = i - period - forecast_horizon
            end = i - forecast_horizon
            X_train = features_matrix[start:end]
            y_train = fwd_ret[start:end]

            valid = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
            if np.sum(valid) < max(10, k + 1):
                continue
            X_t = X_train[valid]
            y_t = y_train[valid]

            std_w = np.std(X_t, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean_x = np.mean(X_t, axis=0)
            std_x = std_w
            X_t = (X_t - mean_x) / std_x

            model = Ridge(alpha=1.0)
            model.fit(X_t, y_t)

        if model is not None and mean_x is not None and std_x is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed = ((row - mean_x) / std_x).reshape(1, -1)
            predictions[i] = model.predict(normed)[0]
            r_squared[i] = model.score(
                (features_matrix[i - period : i] - mean_x) / std_x,
                fwd_ret[i - period : i],
            ) if not np.any(np.isnan(fwd_ret[i - period : i])) else np.nan

    return predictions, r_squared
