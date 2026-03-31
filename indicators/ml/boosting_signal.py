import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


def gradient_boost_signal(
    closes: np.ndarray,
    features_matrix: np.ndarray,
    period: int = 120,
    n_estimators: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling gradient boosting direction prediction.

    Trains a GBT classifier on trailing ``period`` bars to predict whether
    the next-bar return is positive.  Retrains every ``period // 4`` bars.

    Parameters
    ----------
    closes : (N,) price series.
    features_matrix : (N, K) feature array.
    period : training window length.
    n_estimators : number of boosting rounds.

    Returns
    -------
    signal : (N,) predicted probability of positive return (0-1).
    importance_top3 : (N, 3) feature indices of top-3 most important features.
    """
    n = len(closes)
    k = features_matrix.shape[1]
    signal = np.full(n, np.nan, dtype=np.float64)
    importance_top3 = np.full((n, 3), np.nan, dtype=np.float64)

    if n < period + 2:
        return signal, importance_top3

    # Forward 1-bar return direction
    safe = np.maximum(closes, 1e-12)
    fwd_dir = np.full(n, np.nan, dtype=np.float64)
    fwd_dir[:-1] = np.where(safe[1:] > safe[:-1], 1.0, 0.0)

    retrain_every = max(1, period // 4)
    model: GradientBoostingClassifier | None = None
    mean_x: np.ndarray | None = None
    std_x: np.ndarray | None = None
    top3_idx: np.ndarray | None = None

    for i in range(period + 1, n):
        need_train = model is None or (i - period - 1) % retrain_every == 0

        if need_train:
            start = i - period - 1
            end = i - 1
            X_train = features_matrix[start:end]
            y_train = fwd_dir[start:end]

            valid = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
            if np.sum(valid) < max(20, k + 1):
                continue
            X_t = X_train[valid]
            y_t = y_train[valid]

            # Need both classes
            if len(np.unique(y_t)) < 2:
                continue

            std_w = np.std(X_t, axis=0)
            if np.any(std_w < 1e-12):
                continue
            mean_x = np.mean(X_t, axis=0)
            std_x = std_w
            X_t = (X_t - mean_x) / std_x

            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )
            model.fit(X_t, y_t)

            # Feature importance top 3
            imp = model.feature_importances_
            top3_idx = np.argsort(imp)[::-1][:3]
            if len(top3_idx) < 3:
                top3_idx = np.pad(top3_idx, (0, 3 - len(top3_idx)),
                                  constant_values=-1)

        if model is not None and mean_x is not None and std_x is not None:
            row = features_matrix[i]
            if np.any(np.isnan(row)):
                continue
            normed = ((row - mean_x) / std_x).reshape(1, -1)
            prob = model.predict_proba(normed)[0]
            # Probability of class 1.0
            cls_idx = list(model.classes_).index(1.0) if 1.0 in model.classes_ else -1
            if cls_idx >= 0:
                signal[i] = prob[cls_idx]
            if top3_idx is not None:
                importance_top3[i] = top3_idx.astype(np.float64)

    return signal, importance_top3
