import numpy as np


def pvt_strength(closes: np.ndarray, volumes: np.ndarray,
                 period: int = 20) -> tuple:
    """Price-Volume trend strength.

    Measures whether price trend is confirmed by volume. Strong trends
    have price and volume moving together (up-moves with high volume).

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    volumes : np.ndarray
        Trading volumes.
    period : int
        Rolling window for correlation and scoring.

    Returns
    -------
    pvt_score : np.ndarray
        Price-volume trend strength score in [-1, 1].
        Positive = trend confirmed by volume.
    is_confirmed : np.ndarray (bool)
        True if pvt_score > 0.3 (volume confirms trend).
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    pvt_score = np.full(n, np.nan)
    is_confirmed = np.zeros(n, dtype=bool)

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    for i in range(period, n):
        r_win = rets[i - period + 1:i + 1]
        v_win = volumes[i - period + 1:i + 1]

        valid = np.isfinite(r_win) & np.isfinite(v_win) & (v_win > 0)
        if np.sum(valid) < period // 2:
            continue

        r_v = r_win[valid]
        v_v = v_win[valid]

        # Normalize volume
        v_mean = np.mean(v_v)
        if v_mean <= 0:
            continue
        v_norm = v_v / v_mean

        # Volume-weighted directional score:
        # positive returns with above-avg volume = bullish confirmation
        # negative returns with above-avg volume = bearish confirmation
        weighted_dir = np.sum(r_v * v_norm) / len(r_v)

        # Also compute correlation between |returns| and volume
        abs_r = np.abs(r_v)
        r_std = np.std(abs_r)
        v_std = np.std(v_norm)
        if r_std > 1e-9 and v_std > 1e-9:
            corr = np.corrcoef(abs_r, v_norm)[0, 1]
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0

        # Combine: direction * volume confirmation
        score = np.clip(weighted_dir * 100.0 * (0.5 + 0.5 * corr), -1.0, 1.0)
        pvt_score[i] = score
        is_confirmed[i] = abs(score) > 0.3

    return pvt_score, is_confirmed
