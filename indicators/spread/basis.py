"""Futures basis (front minus back contract) and basis momentum.

Measures contango/backwardation and how fast the term structure is
shifting.
"""

import numpy as np


def basis(
    front_closes: np.ndarray,
    back_closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Futures basis and basis momentum.

    Parameters
    ----------
    front_closes : near-month (front) contract closing prices.
    back_closes  : far-month (back) contract closing prices.
    period       : lookback for basis momentum (ROC of basis).

    Returns
    -------
    (basis_val, basis_pct, basis_momentum)
        basis_val      – front - back (positive = backwardation).
        basis_pct      – basis as percentage of back price.
        basis_momentum – change in basis_pct over *period* bars.
    """
    n = len(front_closes)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    basis_val = front_closes - back_closes

    safe_back = np.where(back_closes == 0, np.nan, back_closes)
    basis_pct = basis_val / safe_back * 100.0

    basis_momentum = np.full(n, np.nan)
    for i in range(period, n):
        prev = basis_pct[i - period]
        if not np.isnan(prev):
            basis_momentum[i] = basis_pct[i] - prev

    return basis_val, basis_pct, basis_momentum
