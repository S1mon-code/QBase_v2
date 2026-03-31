import numpy as np


def dtw_distance(
    series_a: np.ndarray,
    series_b: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Dynamic Time Warping distance between two series.

    Computes DTW distance on trailing windows of length ``period``.
    High distance = patterns diverging.

    Parameters
    ----------
    series_a : (N,) first series.
    series_b : (N,) second series.
    period : rolling window length.

    Returns
    -------
    dtw_dist : (N,) DTW distance.
    dtw_zscore : (N,) z-score of DTW distance over expanding history.
    """
    n = len(series_a)
    dtw_dist = np.full(n, np.nan, dtype=np.float64)
    dtw_zscore = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return dtw_dist, dtw_zscore

    # Collect distances for z-score
    dist_history = []

    for i in range(period, n):
        wa = series_a[i - period : i]
        wb = series_b[i - period : i]
        if np.any(np.isnan(wa)) or np.any(np.isnan(wb)):
            continue

        # Z-normalise windows
        sa = np.std(wa)
        sb = np.std(wb)
        if sa < 1e-12 or sb < 1e-12:
            continue
        wa_n = (wa - np.mean(wa)) / sa
        wb_n = (wb - np.mean(wb)) / sb

        d = _dtw_cost(wa_n, wb_n)
        dtw_dist[i] = d
        dist_history.append(d)

        if len(dist_history) >= 2:
            mu = np.mean(dist_history)
            sigma = np.std(dist_history)
            if sigma > 1e-12:
                dtw_zscore[i] = (d - mu) / sigma

    return dtw_dist, dtw_zscore


def _dtw_cost(a: np.ndarray, b: np.ndarray) -> float:
    """Compute DTW distance with Sakoe-Chiba band for efficiency."""
    m = len(a)
    n = len(b)
    band = max(1, m // 4)

    # Use a banded DTW for speed
    dtw_matrix = np.full((m + 1, n + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, m + 1):
        j_start = max(1, i - band)
        j_end = min(n, i + band)
        for j in range(j_start, j_end + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    return np.sqrt(dtw_matrix[m, n])
