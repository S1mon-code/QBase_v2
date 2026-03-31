"""Carry signal from front/back month price relationship.

Measures roll yield (contango/backwardation) as a trading signal.
Positive carry (backwardation) favours long positions; negative
carry (contango) favours short.
"""

import numpy as np


def carry_signal(
    front_closes: np.ndarray,
    back_closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carry signal from front-back month spread.

    Parameters
    ----------
    front_closes : closing prices of front-month contract.
    back_closes  : closing prices of back-month contract.
    period       : lookback for z-score and momentum.

    Returns
    -------
    (carry, carry_zscore, carry_momentum)
        carry           – annualised carry = (back - front) / front.
                          Positive = backwardation, negative = contango.
        carry_zscore    – rolling z-score of carry.
        carry_momentum  – rate of change of carry over period.
    """
    n = len(front_closes)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy(), empty.copy()

    carry = np.full(n, np.nan)
    for i in range(n):
        if front_closes[i] != 0 and not np.isnan(front_closes[i]) and not np.isnan(back_closes[i]):
            carry[i] = (back_closes[i] - front_closes[i]) / front_closes[i]

    carry_z = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = carry[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma > 0:
            carry_z[i] = (carry[i] - mu) / sigma

    carry_mom = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(carry[i]) and not np.isnan(carry[i - period]):
            carry_mom[i] = carry[i] - carry[i - period]

    return carry, carry_z, carry_mom
