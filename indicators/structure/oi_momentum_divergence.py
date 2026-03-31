import numpy as np


def oi_momentum_price_divergence(closes: np.ndarray, oi: np.ndarray,
                                  period: int = 20) -> tuple:
    """Compare momentum of OI vs momentum of price.

    Divergence between the two signals a potential reversal:
    if price momentum is positive but OI momentum is negative,
    the rally lacks conviction.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    period : int
        Lookback for momentum (ROC) calculation.

    Returns
    -------
    oi_mom : np.ndarray
        OI momentum (rate of change).
    price_mom : np.ndarray
        Price momentum (rate of change).
    divergence_score : np.ndarray
        Difference between normalised price momentum and OI momentum.
        Positive = price rising faster than OI (bearish divergence);
        negative = OI rising faster than price (bullish divergence).
    """
    n = len(closes)
    oi_mom = np.full(n, np.nan)
    price_mom = np.full(n, np.nan)
    divergence_score = np.full(n, np.nan)

    if n <= period:
        return oi_mom, price_mom, divergence_score

    for i in range(period, n):
        p_prev = closes[i - period]
        o_prev = oi[i - period]

        if p_prev > 0:
            price_mom[i] = (closes[i] - p_prev) / p_prev
        if o_prev > 0:
            oi_mom[i] = (oi[i] - o_prev) / o_prev

    # Normalise both and compute divergence
    valid = ~np.isnan(price_mom) & ~np.isnan(oi_mom)
    if np.sum(valid) < 2:
        return oi_mom, price_mom, divergence_score

    for i in range(2 * period, n):
        win_p = price_mom[i - period:i]
        win_o = oi_mom[i - period:i]

        mask = ~np.isnan(win_p) & ~np.isnan(win_o)
        if np.sum(mask) < 3:
            continue

        p_std = np.std(win_p[mask], ddof=1)
        o_std = np.std(win_o[mask], ddof=1)

        p_z = (price_mom[i] - np.mean(win_p[mask])) / p_std if p_std > 0 else 0.0
        o_z = (oi_mom[i] - np.mean(win_o[mask])) / o_std if o_std > 0 else 0.0

        divergence_score[i] = p_z - o_z

    return oi_mom, price_mom, divergence_score
