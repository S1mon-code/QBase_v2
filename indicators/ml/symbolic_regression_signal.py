import numpy as np


def symbolic_features(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    period: int = 60,
) -> np.ndarray:
    """Hand-crafted symbolic features from OHLCV data.

    Computes 8 non-linear feature transformations commonly used in
    symbolic regression / genetic programming approaches.

    Parameters
    ----------
    closes : (N,) close prices.
    highs : (N,) high prices.
    lows : (N,) low prices.
    volumes : (N,) volume.
    period : rolling window for normalization.

    Returns
    -------
    features : (N, 8) symbolic feature matrix.

    Feature list:
        0: log(H/L)            — range intensity
        1: log(C/C_prev)       — log return
        2: V * |ret|           — volume-weighted move
        3: (H-L)/C             — normalized range
        4: (C-L)/(H-L)         — close location value
        5: log(V/V_mean)       — relative volume
        6: (H-L) / sqrt(period)  — scaled range (Parkinson-like)
        7: sign(ret)*log(1+V)  — signed volume impulse
    """
    n = len(closes)
    features = np.full((n, 8), np.nan, dtype=np.float64)

    safe_c = np.maximum(closes, 1e-12)
    safe_h = np.maximum(highs, 1e-12)
    safe_l = np.maximum(lows, 1e-12)
    safe_v = np.maximum(volumes, 1e-12)

    # Feature 0: log(H/L)
    hl_ratio = safe_h / safe_l
    features[:, 0] = np.log(hl_ratio)

    # Feature 1: log(C/C_prev)
    features[1:, 1] = np.log(safe_c[1:] / safe_c[:-1])

    # Feature 2: V * |ret|
    ret = np.full(n, np.nan, dtype=np.float64)
    ret[1:] = safe_c[1:] / safe_c[:-1] - 1.0
    features[:, 2] = safe_v * np.abs(np.where(np.isnan(ret), 0.0, ret))
    features[0, 2] = np.nan

    # Feature 3: (H-L)/C
    features[:, 3] = (highs - lows) / safe_c

    # Feature 4: (C-L)/(H-L) — close location
    hl_diff = highs - lows
    safe_hl = np.where(hl_diff < 1e-12, 1e-12, hl_diff)
    features[:, 4] = (closes - lows) / safe_hl

    # Feature 5: log(V/V_mean) — rolling relative volume
    for i in range(period, n):
        v_mean = np.mean(safe_v[i - period : i])
        if v_mean > 1e-12:
            features[i, 5] = np.log(safe_v[i] / v_mean)

    # Feature 6: (H-L) / sqrt(period) — scaled range
    features[:, 6] = (highs - lows) / np.sqrt(float(period))

    # Feature 7: sign(ret) * log(1+V)
    sign_ret = np.sign(np.where(np.isnan(ret), 0.0, ret))
    features[:, 7] = sign_ret * np.log1p(safe_v)
    features[0, 7] = np.nan

    return features
