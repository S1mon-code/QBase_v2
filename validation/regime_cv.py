"""Layer 1: Regime-Conditional Cross-Validation.

Validates strategy consistency across regime folds. A strategy must
demonstrate positive performance across multiple time segments of
the same regime type to pass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeCVResult:
    """Result of regime-conditional cross-validation."""

    strategy: str
    regime: str
    n_folds: int
    fold_sharpes: tuple[float, ...]
    mean_sharpe: float
    std_sharpe: float
    win_rate: float  # fraction of folds with Sharpe > 0
    verdict: str  # "PASS" / "MARGINAL" / "FAIL"


def regime_cv_verdict(mean_sharpe: float, win_rate: float) -> str:
    """Determine regime CV verdict from mean Sharpe and win rate.

    Returns:
        "PASS" if mean_sharpe > 0.3 and win_rate >= 0.5
        "MARGINAL" if mean_sharpe > 0 and win_rate >= 0.33
        "FAIL" otherwise
    """
    if mean_sharpe > 0.3 and win_rate >= 0.5:
        return "PASS"
    if mean_sharpe > 0.0 and win_rate >= 0.33:
        return "MARGINAL"
    return "FAIL"


def run_regime_cv(
    fold_sharpes: list[float],
    strategy: str = "",
    regime: str = "",
) -> RegimeCVResult:
    """Compute Regime CV result from per-fold Sharpe ratios.

    Args:
        fold_sharpes: Sharpe ratio for each fold.
        strategy: Strategy name for labeling.
        regime: Regime name for labeling.

    Returns:
        RegimeCVResult with computed statistics and verdict.
    """
    n_folds = len(fold_sharpes)

    if n_folds == 0:
        return RegimeCVResult(
            strategy=strategy,
            regime=regime,
            n_folds=0,
            fold_sharpes=(),
            mean_sharpe=0.0,
            std_sharpe=0.0,
            win_rate=0.0,
            verdict="FAIL",
        )

    sharpes_tuple = tuple(fold_sharpes)
    mean_sharpe = sum(fold_sharpes) / n_folds

    if n_folds == 1:
        std_sharpe = 0.0
    else:
        variance = sum((s - mean_sharpe) ** 2 for s in fold_sharpes) / (n_folds - 1)
        std_sharpe = variance**0.5

    win_count = sum(1 for s in fold_sharpes if s > 0.0)
    win_rate = win_count / n_folds

    verdict = regime_cv_verdict(mean_sharpe, win_rate)

    return RegimeCVResult(
        strategy=strategy,
        regime=regime,
        n_folds=n_folds,
        fold_sharpes=sharpes_tuple,
        mean_sharpe=mean_sharpe,
        std_sharpe=std_sharpe,
        win_rate=win_rate,
        verdict=verdict,
    )
