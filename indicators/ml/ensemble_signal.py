import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier


def ensemble_vote(
    closes: np.ndarray,
    features_matrix: np.ndarray,
    period: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble of 3 simple models voting on direction.

    Combines Ridge regression, Lasso regression, and Random Forest
    classifier.  Each model votes +1 (bullish) or -1 (bearish) based
    on its prediction.  Retrains every ``period // 4`` bars.

    Parameters
    ----------
    closes : (N,) price series.
    features_matrix : (N, K) feature array.
    period : training window length.

    Returns
    -------
    vote_score : (N,) averaged vote from -1 to +1.
    agreement : (N,) fraction of models that agree (0.33, 0.67, or 1.0).
    """
    n = len(closes)
    k = features_matrix.shape[1]
    vote_score = np.full(n, np.nan, dtype=np.float64)
    agreement = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return vote_score, agreement

    # 1-bar return
    safe = np.maximum(closes, 1e-12)
    log_p = np.log(safe)
    ret1 = np.full(n, np.nan, dtype=np.float64)
    ret1[:-1] = log_p[1:] - log_p[:-1]
    direction = np.sign(ret1)

    retrain_every = max(1, period // 4)
    ridge_model: Ridge | None = None
    lasso_model: Lasso | None = None
    rf_model: RandomForestClassifier | None = None
    mean_x: np.ndarray | None = None
    std_x: np.ndarray | None = None

    for i in range(period + 1, n):
        need_train = ridge_model is None or (i - period - 1) % retrain_every == 0

        if need_train:
            start = i - period - 1
            end = i - 1
            X_train = features_matrix[start:end]
            y_ret = ret1[start:end]
            y_dir = direction[start:end]

            valid = ~np.isnan(y_ret) & ~np.any(np.isnan(X_train), axis=1)
            if np.sum(valid) < max(20, k + 1):
                continue
            X_t = X_train[valid]
            y_r = y_ret[valid]
            y_d = y_dir[valid]

            unique = np.unique(y_d)
            if len(unique) < 2:
                continue

            std_w = np.std(X_t, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean_x = np.mean(X_t, axis=0)
            std_x = std_w
            X_normed = (X_t - mean_x) / std_x

            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_normed, y_r)

            lasso_model = Lasso(alpha=0.001, max_iter=2000)
            lasso_model.fit(X_normed, y_r)

            rf_model = RandomForestClassifier(
                n_estimators=30, max_depth=4, random_state=42,
            )
            rf_model.fit(X_normed, y_d)

        if ridge_model is not None and mean_x is not None and std_x is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed = ((row - mean_x) / std_x).reshape(1, -1)

            votes = []
            # Ridge vote
            pred_ridge = ridge_model.predict(normed)[0]
            votes.append(1.0 if pred_ridge > 0 else -1.0)
            # Lasso vote
            pred_lasso = lasso_model.predict(normed)[0]
            votes.append(1.0 if pred_lasso > 0 else -1.0)
            # RF vote
            pred_rf = rf_model.predict(normed)[0]
            votes.append(float(pred_rf))

            vote_score[i] = np.mean(votes)
            agreement[i] = np.sum(np.array(votes) == np.sign(np.mean(votes))) / 3.0

    return vote_score, agreement
