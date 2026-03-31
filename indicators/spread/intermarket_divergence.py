"""Intermarket divergence detector.

Detects when two normally correlated markets diverge, signalling
potential mean-reversion opportunity or regime change.
"""

import numpy as np


def intermarket_divergence(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect divergence between two correlated markets.

    Parameters
    ----------
    closes_a : closing prices of asset A.
    closes_b : closing prices of asset B.
    period   : lookback window for correlation and divergence.

    Returns
    -------
    (divergence_score, is_diverging, expected_convergence)
        divergence_score    – difference in normalised returns (z-scored).
        is_diverging        – 1.0 if |divergence_score| > 2, else 0.0.
        expected_convergence – sign of expected convergence direction
                               (+1 = A should outperform, -1 = B should).
    """
    n = len(closes_a)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy(), empty.copy()

    div_score = np.full(n, np.nan)
    is_div = np.full(n, np.nan)
    expected = np.full(n, np.nan)

    ret_a = np.full(n, np.nan)
    ret_b = np.full(n, np.nan)
    for i in range(1, n):
        if closes_a[i - 1] != 0 and not np.isnan(closes_a[i - 1]):
            ret_a[i] = closes_a[i] / closes_a[i - 1] - 1.0
        if closes_b[i - 1] != 0 and not np.isnan(closes_b[i - 1]):
            ret_b[i] = closes_b[i] / closes_b[i - 1] - 1.0

    for i in range(period, n):
        wa = ret_a[i - period + 1 : i + 1]
        wb = ret_b[i - period + 1 : i + 1]
        mask = ~(np.isnan(wa) | np.isnan(wb))
        if np.sum(mask) < 5:
            continue

        cum_a = np.sum(wa[mask])
        cum_b = np.sum(wb[mask])

        std_a = np.std(wa[mask], ddof=1)
        std_b = np.std(wb[mask], ddof=1)
        if std_a == 0 or std_b == 0:
            continue

        norm_a = cum_a / std_a
        norm_b = cum_b / std_b

        div_score[i] = norm_a - norm_b
        is_div[i] = 1.0 if abs(div_score[i]) > 2.0 else 0.0
        # If A outperformed (positive div), expect B to catch up → A should underperform
        expected[i] = -1.0 if div_score[i] > 0 else 1.0

    return div_score, is_div, expected
