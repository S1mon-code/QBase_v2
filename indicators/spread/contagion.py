"""Cross-asset contagion / tail-event spillover detector.

Measures how correlated two assets become during extreme (tail) moves,
which is a different regime from normal-market correlation.
"""

import numpy as np


def contagion_score(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    period: int = 60,
    threshold: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-asset contagion/spillover during extreme moves.

    Parameters
    ----------
    returns_a : return series of asset A.
    returns_b : return series of asset B.
    period    : rolling window for statistics.
    threshold : number of standard deviations to define an extreme move.

    Returns
    -------
    (contagion, exceedance_corr)
        contagion      – fraction of extreme events in A that coincide
                         with extreme events in B (0-1).
        exceedance_corr – correlation of returns computed only on days
                          where at least one asset has an extreme move.
    """
    n = len(returns_a)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty

    contagion = np.full(n, np.nan)
    exc_corr = np.full(n, np.nan)

    for i in range(period - 1, n):
        ra = returns_a[i - period + 1 : i + 1].astype(float)
        rb = returns_b[i - period + 1 : i + 1].astype(float)
        mask = ~(np.isnan(ra) | np.isnan(rb))
        if np.sum(mask) < 10:
            continue
        rav, rbv = ra[mask], rb[mask]

        std_a = np.std(rav, ddof=1)
        std_b = np.std(rbv, ddof=1)
        if std_a == 0 or std_b == 0:
            continue

        extreme_a = np.abs(rav) > threshold * std_a
        extreme_b = np.abs(rbv) > threshold * std_b

        n_extreme_a = np.sum(extreme_a)
        if n_extreme_a > 0:
            contagion[i] = np.sum(extreme_a & extreme_b) / n_extreme_a

        # Exceedance correlation: on days with any extreme move
        any_extreme = extreme_a | extreme_b
        if np.sum(any_extreme) >= 3:
            ea, eb = rav[any_extreme], rbv[any_extreme]
            sa = np.std(ea, ddof=1)
            sb = np.std(eb, ddof=1)
            if sa > 0 and sb > 0:
                exc_corr[i] = np.corrcoef(ea, eb)[0, 1]

    return contagion, exc_corr
