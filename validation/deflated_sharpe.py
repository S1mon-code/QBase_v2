"""Layer 4: Deflated Sharpe Ratio.

Corrects for multiple testing bias using Bailey & Lopez de Prado (2014).
A high observed Sharpe may be a statistical artifact when many strategy
variants have been tried.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def expected_max_sharpe(
    n_trials: int,
    sharpe_std: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Expected maximum Sharpe ratio given N independent trials.

    Bailey & Lopez de Prado (2014). Uses the Euler-Mascheroni constant
    gamma ~ 0.5772 and the approximation:

        E[max] = sharpe_std * ((1-gamma)*Phi^{-1}(1-1/N) + gamma*Phi^{-1}(1-1/(N*e)))

    Args:
        n_trials: Number of independent strategy trials.
        sharpe_std: Standard deviation of Sharpe ratios across trials.
        skew: Skewness of returns (default 0.0 for normal).
        kurt: Kurtosis of returns (default 3.0 for normal).

    Returns:
        Expected maximum Sharpe ratio.
    """
    if n_trials <= 0:
        return 0.0
    if n_trials == 1:
        return 0.0

    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    e = math.e

    # Avoid numerical issues with very large N
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))

    return sharpe_std * ((1.0 - gamma) * z1 + gamma * z2)


def sharpe_std_error(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Standard error of the Sharpe ratio.

    Lo (2002) with skewness and kurtosis adjustment:

        SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / (n-1))

    Args:
        sharpe: Observed Sharpe ratio.
        n_obs: Number of observations.
        skew: Skewness of returns.
        kurt: Kurtosis of returns.

    Returns:
        Standard error of the Sharpe ratio.
    """
    if n_obs <= 1:
        return math.inf

    sr2 = sharpe * sharpe
    numerator = 1.0 + 0.5 * sr2 - skew * sharpe + (kurt - 3.0) / 4.0 * sr2
    # Ensure non-negative before sqrt
    numerator = max(numerator, 0.0)

    return math.sqrt(numerator / (n_obs - 1))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    sharpe_std: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR).

    DSR = Phi((observed - E[max]) / SE(SR))

    Returns the probability that the true Sharpe ratio is greater than zero
    after correcting for multiple testing.

    Args:
        observed_sharpe: The best observed Sharpe ratio.
        n_trials: Total number of strategy trials attempted.
        sharpe_std: Standard deviation of Sharpe ratios across trials.
        n_obs: Number of return observations.
        skew: Skewness of returns.
        kurt: Kurtosis of returns.

    Returns:
        Probability (0 to 1) that true SR > 0.
    """
    e_max = expected_max_sharpe(n_trials, sharpe_std, skew, kurt)
    se = sharpe_std_error(observed_sharpe, n_obs, skew, kurt)

    if se == 0.0 or math.isinf(se):
        return 0.0

    z = (observed_sharpe - e_max) / se
    return float(stats.norm.cdf(z))
