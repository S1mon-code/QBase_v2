"""Layer 5b: Permutation Test.

Tests whether a strategy's Sharpe ratio is significantly better than
what could be achieved by random chance. Shuffles the return series
and re-evaluates to build a null distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from validation.thresholds import get_thresholds
from validation.utils import compute_sharpe as _compute_sharpe


@dataclass(frozen=True)
class PermutationResult:
    """Result of permutation test."""

    n_permutations: int
    real_sharpe: float
    p_value: float  # fraction of random sharpes >= real sharpe
    verdict: str  # "SIGNIFICANT" / "MARGINAL" / "NOT_SIGNIFICANT"


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

    cfg = get_thresholds()["permutation"]
    if p_value < cfg["significant_pvalue"]:
        verdict = "SIGNIFICANT"
    elif p_value < cfg["marginal_pvalue"]:
        verdict = "MARGINAL"
    else:
        verdict = "NOT_SIGNIFICANT"

    return PermutationResult(
        n_permutations=n_perms,
        real_sharpe=strategy_sharpe,
        p_value=p_value,
        verdict=verdict,
    )
