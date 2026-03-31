"""Layer 5a: Bootstrap Resampling.

Tests result stability by resampling daily returns with replacement.
A robust strategy should maintain positive Sharpe across bootstrap samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Result of bootstrap resampling test."""

    n_simulations: int
    sharpe_ci_lower: float  # 2.5th percentile
    sharpe_ci_upper: float  # 97.5th percentile
    sharpe_mean: float
    maxdd_median: float
    maxdd_95th: float
    verdict: str  # "ROBUST" / "ACCEPTABLE" / "FRAGILE"


def _compute_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if len(returns) == 0:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0.0:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(252))


def _compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from daily returns."""
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(np.min(drawdowns))


def bootstrap_test(
    daily_returns: np.ndarray,
    n_sims: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap resampling of daily returns.

    Resamples the daily return series with replacement n_sims times,
    computing Sharpe ratio and max drawdown for each sample.

    Verdict:
        - "ROBUST": 95% CI lower bound > 0
        - "ACCEPTABLE": mean Sharpe > 0 but CI crosses zero
        - "FRAGILE": mean Sharpe <= 0

    Args:
        daily_returns: Array of daily returns.
        n_sims: Number of bootstrap simulations.
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with confidence intervals and verdict.
    """
    rng = np.random.default_rng(seed)
    n = len(daily_returns)

    if n == 0:
        return BootstrapResult(
            n_simulations=n_sims,
            sharpe_ci_lower=0.0,
            sharpe_ci_upper=0.0,
            sharpe_mean=0.0,
            maxdd_median=0.0,
            maxdd_95th=0.0,
            verdict="FRAGILE",
        )

    sharpes = np.empty(n_sims)
    maxdds = np.empty(n_sims)

    for i in range(n_sims):
        indices = rng.integers(0, n, size=n)
        sample = daily_returns[indices]
        sharpes[i] = _compute_sharpe(sample)
        maxdds[i] = _compute_max_drawdown(sample)

    sharpe_ci_lower = float(np.percentile(sharpes, 2.5))
    sharpe_ci_upper = float(np.percentile(sharpes, 97.5))
    sharpe_mean = float(np.mean(sharpes))
    maxdd_median = float(np.median(maxdds))
    maxdd_95th = float(np.percentile(maxdds, 95))

    if sharpe_ci_lower > 0.0:
        verdict = "ROBUST"
    elif sharpe_mean > 0.0:
        verdict = "ACCEPTABLE"
    else:
        verdict = "FRAGILE"

    return BootstrapResult(
        n_simulations=n_sims,
        sharpe_ci_lower=sharpe_ci_lower,
        sharpe_ci_upper=sharpe_ci_upper,
        sharpe_mean=sharpe_mean,
        maxdd_median=maxdd_median,
        maxdd_95th=maxdd_95th,
        verdict=verdict,
    )
