import numpy as np


def composite_regime(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite trend/range regime score combining ADX-like + volatility ratio + autocorrelation.

    Blends three regime signals:
    1. Directional movement ratio (ADX-like)
    2. Efficiency ratio (net move / total path)
    3. Return autocorrelation at lag 1

    Returns (regime_score, is_trending, is_ranging).
    regime_score: -1 (strong range) to +1 (strong trend).
    is_trending = 1 when regime_score > 0.3.
    is_ranging = 1 when regime_score < -0.3.
    """
    n = len(closes)
    regime_score = np.full(n, np.nan, dtype=np.float64)
    is_trending = np.full(n, np.nan, dtype=np.float64)
    is_ranging = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return regime_score, is_trending, is_ranging

    # --- Component 1: Efficiency Ratio ---
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        net_move = abs(closes[i] - closes[i - period])
        total_move = np.sum(np.abs(np.diff(closes[i - period : i + 1])))
        if total_move > 1e-14:
            er[i] = net_move / total_move
        else:
            er[i] = 0.0

    # --- Component 2: Volatility Ratio (high-low range vs close-to-close) ---
    vr = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        hl_range = np.mean(highs[i - period : i] - lows[i - period : i])
        cc_range = np.std(np.diff(closes[i - period : i + 1]), ddof=1)
        if cc_range > 1e-14:
            # High vr = noisy (ranging), low vr = directional
            vr[i] = cc_range / (hl_range + 1e-14)
        else:
            vr[i] = 0.0

    # --- Component 3: Return Autocorrelation (lag 1) ---
    ac = np.full(n, np.nan, dtype=np.float64)
    safe = np.maximum(closes, 1e-9)
    log_ret = np.diff(np.log(safe))

    for i in range(period + 1, n):
        window = log_ret[i - period : i]
        mu = np.mean(window)
        var = np.var(window, ddof=0)
        if var < 1e-14:
            ac[i] = 0.0
            continue
        centered = window - mu
        w_len = len(centered)
        ac[i] = np.dot(centered[:w_len - 1], centered[1:]) / (var * w_len)

    # --- Combine into regime score ---
    for i in range(period + 1, n):
        if np.isnan(er[i]) or np.isnan(vr[i]) or np.isnan(ac[i]):
            continue

        # ER: 0 (noise) to 1 (perfect trend) -> map to [-1, 1]
        er_score = 2.0 * er[i] - 1.0

        # VR: high = trending signal is strong relative to noise -> map
        vr_score = np.clip(2.0 * vr[i] - 1.0, -1.0, 1.0)

        # AC: positive = trending, negative = mean reverting, already ~[-1, 1]
        ac_score = np.clip(ac[i] * 3.0, -1.0, 1.0)

        # Weighted blend: ER most important, AC second, VR third
        score = 0.5 * er_score + 0.3 * ac_score + 0.2 * vr_score
        score = float(np.clip(score, -1.0, 1.0))

        regime_score[i] = score
        is_trending[i] = 1.0 if score > 0.3 else 0.0
        is_ranging[i] = 1.0 if score < -0.3 else 0.0

    return regime_score, is_trending, is_ranging
