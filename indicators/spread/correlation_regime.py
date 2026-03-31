"""Rolling correlation regime change detector.

Computes fast and slow rolling correlations between two assets and
flags divergences that may signal a regime shift.
"""

import numpy as np


def correlation_regime(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    fast: int = 20,
    slow: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling correlation regime change detector.

    Parameters
    ----------
    closes_a : close prices of asset A.
    closes_b : close prices of asset B.
    fast     : fast rolling correlation window.
    slow     : slow rolling correlation window.

    Returns
    -------
    (corr_fast, corr_slow, corr_divergence)
        corr_fast       – fast-window rolling Pearson correlation.
        corr_slow       – slow-window rolling Pearson correlation.
        corr_divergence – fast - slow.  Large divergence signals a
                          regime shift in inter-asset relationships.
    """
    n = len(closes_a)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    # Log returns
    ret_a = np.full(n, np.nan)
    ret_b = np.full(n, np.nan)
    safe_a = np.where(closes_a == 0, np.nan, closes_a)
    safe_b = np.where(closes_b == 0, np.nan, closes_b)
    ret_a[1:] = np.log(safe_a[1:] / safe_a[:-1])
    ret_b[1:] = np.log(safe_b[1:] / safe_b[:-1])

    corr_fast = np.full(n, np.nan)
    corr_slow = np.full(n, np.nan)

    def _rolling_corr(ra: np.ndarray, rb: np.ndarray, window: int, out: np.ndarray) -> None:
        for i in range(window, n):
            wa = ra[i - window + 1 : i + 1]
            wb = rb[i - window + 1 : i + 1]
            mask = ~(np.isnan(wa) | np.isnan(wb))
            if np.sum(mask) < 5:
                continue
            va, vb = wa[mask], wb[mask]
            std_a = np.std(va, ddof=1)
            std_b = np.std(vb, ddof=1)
            if std_a > 0 and std_b > 0:
                out[i] = np.corrcoef(va, vb)[0, 1]

    _rolling_corr(ret_a, ret_b, fast, corr_fast)
    _rolling_corr(ret_a, ret_b, slow, corr_slow)

    corr_divergence = corr_fast - corr_slow

    return corr_fast, corr_slow, corr_divergence
