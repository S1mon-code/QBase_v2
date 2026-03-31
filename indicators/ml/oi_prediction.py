import numpy as np
from sklearn.linear_model import Ridge


def oi_predicted_return(closes: np.ndarray, oi: np.ndarray,
                        volumes: np.ndarray, period: int = 120) -> tuple:
    """Ridge regression: predict next-bar return from OI features.

    Uses a rolling window to fit a Ridge regression model that
    predicts the next bar's return from OI-derived features.
    Retrained every period//4 bars.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volume.
    period : int
        Training window size.

    Returns
    -------
    predicted_return : np.ndarray
        Predicted next-bar return.
    r_squared : np.ndarray
        In-sample R-squared of the current model.
    """
    n = len(closes)
    predicted_return = np.full(n, np.nan)
    r_squared = np.full(n, np.nan)

    if n < period + 2:
        return predicted_return, r_squared

    # Pre-compute features
    ret = np.full(n, 0.0)
    ret[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-10)

    oi_chg = np.full(n, 0.0)
    oi_chg[1:] = np.diff(oi)

    oi_chg_pct = np.full(n, 0.0)
    for i in range(1, n):
        if oi[i - 1] > 0:
            oi_chg_pct[i] = oi_chg[i] / oi[i - 1]

    vol_oi = np.full(n, 0.0)
    for i in range(n):
        if oi[i] > 0:
            vol_oi[i] = volumes[i] / oi[i]

    retrain_interval = max(period // 4, 1)
    model = None
    current_r2 = np.nan
    feat_mean = None
    feat_std = None

    for i in range(period + 1, n):
        if model is None or (i - period - 1) % retrain_interval == 0:
            idx_s = i - period - 1
            idx_e = i - 1  # leave last bar for prediction

            # Features: oi_chg_pct, vol_oi, lagged return
            X = np.column_stack([
                oi_chg_pct[idx_s:idx_e],
                vol_oi[idx_s:idx_e],
                ret[idx_s:idx_e],
            ])
            # Target: next-bar return
            y = ret[idx_s + 1:idx_e + 1]

            valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
            X_valid = X[valid]
            y_valid = y[valid]

            if len(X_valid) < 20:
                continue

            feat_mean = np.mean(X_valid, axis=0)
            feat_std = np.std(X_valid, axis=0)
            feat_std[feat_std == 0] = 1.0
            X_norm = (X_valid - feat_mean) / feat_std

            model = Ridge(alpha=1.0)
            model.fit(X_norm, y_valid)

            y_pred_train = model.predict(X_norm)
            ss_res = np.sum((y_valid - y_pred_train) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            current_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if model is None or feat_mean is None:
            continue

        feat = np.array([
            oi_chg_pct[i], vol_oi[i], ret[i]
        ]).reshape(1, -1)
        feat_norm = (feat - feat_mean) / feat_std

        predicted_return[i] = model.predict(feat_norm)[0]
        r_squared[i] = current_r2

    return predicted_return, r_squared
