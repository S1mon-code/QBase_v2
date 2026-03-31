import numpy as np
from sklearn.linear_model import Ridge


def cv_signal_strength(closes, features_matrix, period=120, n_folds=3):
    """Rolling cross-validated signal strength: how predictable are returns
    from features?

    Within each rolling window, performs time-series-aware k-fold
    cross-validation of a Ridge regression predicting next-bar returns
    from features.  Retrained every ``period // 4`` bars.

    Parameters
    ----------
    closes : 1-D array of close prices.
    features_matrix : (N, K) array of features.
    period : rolling window size for training.
    n_folds : number of CV folds (time-series split).

    Returns
    -------
    cv_r_squared : (N,) average out-of-fold R^2.  Positive = features are
        predictive beyond random.
    signal_strength : (N,) clipped and normalised version in [0, 1].
    """
    closes = np.asarray(closes, dtype=np.float64)
    features_matrix = np.asarray(features_matrix, dtype=np.float64)
    n, k = features_matrix.shape
    cv_r_squared = np.full(n, np.nan, dtype=np.float64)
    signal_strength = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return cv_r_squared, signal_strength

    # forward returns (1-bar ahead) -- computed causally
    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    fwd_ret[:-1] = closes[1:] / np.maximum(closes[:-1], 1e-10) - 1

    retrain_interval = max(1, period // 4)
    last_r2 = np.nan

    for i in range(period, n):
        need_retrain = np.isnan(last_r2) or ((i - period) % retrain_interval == 0)

        if need_retrain:
            # window: [i-period, i-1] -- features predict fwd_ret
            # fwd_ret[t] = return from t to t+1, so we need features[t] and fwd_ret[t]
            # but fwd_ret[t] uses closes[t+1], which is available at bar t+1
            # For the window ending at i-1, fwd_ret[i-2] uses closes[i-1] which is known
            end_idx = i - 1  # last bar whose fwd_ret is known
            start_idx = end_idx - period + 1

            if start_idx < 0:
                continue

            X = features_matrix[start_idx: end_idx + 1]
            y = fwd_ret[start_idx: end_idx + 1]

            # remove NaN rows
            valid = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
            X_clean = X[valid]
            y_clean = y[valid]

            if len(y_clean) < n_folds * 5:
                continue

            # time-series CV
            fold_size = len(y_clean) // n_folds
            r2_scores = []

            for fold in range(1, n_folds):
                train_end = fold * fold_size
                val_start = train_end
                val_end = min(val_start + fold_size, len(y_clean))

                if val_end <= val_start or train_end < k + 2:
                    continue

                X_train = X_clean[:train_end]
                y_train = y_clean[:train_end]
                X_val = X_clean[val_start: val_end]
                y_val = y_clean[val_start: val_end]

                # standardise
                mu_x = X_train.mean(axis=0)
                std_x = X_train.std(axis=0)
                std_x[std_x < 1e-10] = 1.0

                X_tr_norm = (X_train - mu_x) / std_x
                X_vl_norm = (X_val - mu_x) / std_x

                try:
                    model = Ridge(alpha=1.0)
                    model.fit(X_tr_norm, y_train)
                    y_pred = model.predict(X_vl_norm)
                except Exception:
                    continue

                ss_res = np.sum((y_val - y_pred) ** 2)
                ss_tot = np.sum((y_val - y_val.mean()) ** 2)
                if ss_tot < 1e-15:
                    continue
                r2 = 1 - ss_res / ss_tot
                r2_scores.append(r2)

            if len(r2_scores) > 0:
                last_r2 = np.mean(r2_scores)
            else:
                last_r2 = 0.0

        cv_r_squared[i] = last_r2
        # signal strength: clip R^2 to [0, 0.3] and normalise to [0, 1]
        signal_strength[i] = np.clip(last_r2, 0, 0.3) / 0.3

    return cv_r_squared, signal_strength
