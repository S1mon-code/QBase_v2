"""Layer 5b: Permutation Test.

Tests whether a strategy's Sharpe ratio is significantly better than
what could be achieved by random chance. Shuffles the return series
and re-evaluates to build a null distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PermutationResult:
    """Result of permutation test."""

    n_permutations: int
    real_sharpe: float
    p_value: float  # fraction of random sharpes >= real sharpe
    verdict: str  # "SIGNIFICANT" / "MARGINAL" / "NOT_SIGNIFICANT"


def _compute_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if len(returns) == 0:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0.0:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(252))


def permutation_test(
    daily_returns: np.ndarray,
    strategy_sharpe: float,
    n_perms: int = 1000,
    seed: int = 42,
) -> PermutationResult:
    """Permutation test for strategy signal significance.

    Shuffles daily returns n_perms times, computes Sharpe for each
    permuted series, and calculates the p-value as the fraction of
    permuted Sharpes that equal or exceed the real Sharpe.

    Verdict:
        - "SIGNIFICANT": p < 0.05
        - "MARGINAL": 0.05 <= p < 0.10
        - "NOT_SIGNIFICANT": p >= 0.10

    Args:
        daily_returns: Array of daily returns.
        strategy_sharpe: The observed strategy Sharpe ratio.
        n_perms: Number of permutations.
        seed: Random seed for reproducibility.

    Returns:
        PermutationResult with p-value and verdict.
    """
    rng = np.random.default_rng(seed)

    if len(daily_returns) == 0:
        return PermutationResult(
            n_permutations=n_perms,
            real_sharpe=strategy_sharpe,
            p_value=1.0,
            verdict="NOT_SIGNIFICANT",
        )

    random_sharpes = np.empty(n_perms)
    for i in range(n_perms):
        shuffled = rng.permutation(daily_returns)
        random_sharpes[i] = _compute_sharpe(shuffled)

    p_value = float(np.mean(random_sharpes >= strategy_sharpe))

    if p_value < 0.05:
        verdict = "SIGNIFICANT"
    elif p_value < 0.10:
        verdict = "MARGINAL"
    else:
        verdict = "NOT_SIGNIFICANT"

    return PermutationResult(
        n_permutations=n_perms,
        real_sharpe=strategy_sharpe,
        p_value=p_value,
        verdict=verdict,
    )
