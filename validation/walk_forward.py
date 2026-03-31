"""Layer 3: Walk-Forward Validation.

Tests parameter stability over time using rolling, expanding,
or regime-aware walk-forward windows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardResult:
    """Result of walk-forward validation."""

    mode: str  # "rolling", "expanding", "regime_aware"
    n_windows: int
    window_sharpes: tuple[float, ...]
    mean_sharpe: float
    win_rate: float  # fraction > 0
    worst_sharpe: float
    best_sharpe: float
    passed: bool  # win_rate >= 0.5 and mean_sharpe > 0


def walk_forward_verdict(
    window_sharpes: list[float],
    mode: str = "rolling",
) -> WalkForwardResult:
    """Compute walk-forward result from per-window OOS Sharpe ratios.

    Args:
        window_sharpes: Sharpe ratio for each walk-forward window.
        mode: Walk-forward mode ("rolling", "expanding", "regime_aware").

    Returns:
        WalkForwardResult with aggregated statistics and pass/fail.
    """
    n_windows = len(window_sharpes)

    if n_windows == 0:
        return WalkForwardResult(
            mode=mode,
            n_windows=0,
            window_sharpes=(),
            mean_sharpe=0.0,
            win_rate=0.0,
            worst_sharpe=0.0,
            best_sharpe=0.0,
            passed=False,
        )

    sharpes_tuple = tuple(window_sharpes)
    mean_sharpe = sum(window_sharpes) / n_windows
    win_count = sum(1 for s in window_sharpes if s > 0.0)
    win_rate = win_count / n_windows
    worst_sharpe = min(window_sharpes)
    best_sharpe = max(window_sharpes)
    passed = win_rate >= 0.5 and mean_sharpe > 0.0

    return WalkForwardResult(
        mode=mode,
        n_windows=n_windows,
        window_sharpes=sharpes_tuple,
        mean_sharpe=mean_sharpe,
        win_rate=win_rate,
        worst_sharpe=worst_sharpe,
        best_sharpe=best_sharpe,
        passed=passed,
    )
