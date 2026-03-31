import numpy as np
from sklearn.ensemble import IsolationForest


def oi_anomaly(closes: np.ndarray, oi: np.ndarray,
               volumes: np.ndarray, period: int = 120) -> tuple:
    """Isolation Forest anomaly detection on OI behaviour.

    Detects unusual OI patterns that may precede significant market
    events (large moves, squeezes, etc.).  Retrained every
    period//4 bars.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volume.
    period : int
        Lookback for training the Isolation Forest.

    Returns
    -------
    anomaly_score : np.ndarray
        Anomaly score from Isolation Forest (lower = more anomalous).
    is_anomalous : np.ndarray (float)
        1.0 for detected anomalies, 0.0 otherwise.
    """
    n = len(closes)
    anomaly_score = np.full(n, np.nan)
    is_anomalous = np.zeros(n, dtype=float)

    if n < period + 1:
        return anomaly_score, is_anomalous

    # Pre-compute features
    oi_chg = np.full(n, 0.0)
    oi_chg[1:] = np.diff(oi)

    price_ret = np.full(n, 0.0)
    price_ret[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-10)

    vol_oi = np.full(n, 0.0)
    for i in range(n):
        if oi[i] > 0:
            vol_oi[i] = volumes[i] / oi[i]

    retrain_interval = max(period // 4, 1)
    model = None
    feat_mean = None
    feat_std = None

    for i in range(period, n):
        if model is None or (i - period) % retrain_interval == 0:
            idx_s = i - period
            idx_e = i

            X = np.column_stack([
                oi_chg[idx_s:idx_e],
                price_ret[idx_s:idx_e],
                vol_oi[idx_s:idx_e],
                oi[idx_s:idx_e],
            ])

            valid = ~np.any(np.isnan(X), axis=1)
            X_valid = X[valid]

            if len(X_valid) < 20:
                continue

            feat_mean = np.mean(X_valid, axis=0)
            feat_std = np.std(X_valid, axis=0)
            feat_std[feat_std == 0] = 1.0
            X_norm = (X_valid - feat_mean) / feat_std

            model = IsolationForest(
                n_estimators=50, contamination=0.05,
                random_state=42, max_samples=min(100, len(X_norm))
            )
            model.fit(X_norm)

        if model is None or feat_mean is None:
            continue

        feat = np.array([
            oi_chg[i], price_ret[i], vol_oi[i], oi[i]
        ]).reshape(1, -1)
        feat_norm = (feat - feat_mean) / feat_std

        score = model.score_samples(feat_norm)[0]
        anomaly_score[i] = score

        pred = model.predict(feat_norm)[0]
        if pred == -1:
            is_anomalous[i] = 1.0

    return anomaly_score, is_anomalous
