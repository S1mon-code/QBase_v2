"""Walk-Forward validation for portfolio construction process.

Validates that the strategy selection + weighting process itself is not overfit.

Method:
    1. Split training data into K rolling windows
    2. At each window boundary, run the full strategy selection process
    3. Record which strategies are selected and the resulting OOS performance
    4. Check: are the selected strategies stable across windows?
    5. Check: does the OOS portfolio perform consistently?

A robust portfolio construction process should:
    - Select largely the same strategies across windows (stability > 60%)
    - Show positive OOS performance in > 60% of windows
    - Not have extreme variance in OOS metrics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WFPortfolioResult:
    """Result of walk-forward portfolio validation.

    Attributes
    ----------
    n_windows : int
        Number of WF windows tested.
    selection_stability : float
        Fraction of strategies that appear in > 50% of windows (0-1).
    core_strategies : list[str]
        Strategies selected in > 50% of windows.
    peripheral_strategies : list[str]
        Strategies selected in <= 50% of windows.
    oos_win_rate : float
        Fraction of windows with positive OOS Sharpe.
    oos_sharpes : list[float]
        Per-window OOS Sharpe ratios.
    oos_returns : list[float]
        Per-window OOS returns.
    mean_oos_sharpe : float
        Average OOS Sharpe across windows.
    sharpe_consistency : float
        1 - CV of OOS Sharpes (higher = more consistent).
    verdict : str
        "ROBUST", "MARGINAL", or "FRAGILE".
    """

    n_windows: int
    selection_stability: float
    core_strategies: list[str]
    peripheral_strategies: list[str]
    oos_win_rate: float
    oos_sharpes: list[float]
    oos_returns: list[float]
    mean_oos_sharpe: float
    sharpe_consistency: float
    verdict: str


def validate_portfolio_construction(
    strategy_keys: list[str],
    strategy_sharpes_per_window: list[dict[str, float]],
    selected_per_window: list[list[str]],
    portfolio_oos_sharpes: list[float],
    portfolio_oos_returns: list[float],
) -> WFPortfolioResult:
    """Evaluate walk-forward stability of portfolio construction.

    Parameters
    ----------
    strategy_keys : list[str]
        All candidate strategy names.
    strategy_sharpes_per_window : list[dict[str, float]]
        Per-window dict of {strategy: oos_sharpe} for all candidates.
    selected_per_window : list[list[str]]
        Which strategies were selected in each window.
    portfolio_oos_sharpes : list[float]
        OOS Sharpe of the constructed portfolio per window.
    portfolio_oos_returns : list[float]
        OOS return of the constructed portfolio per window.

    Returns
    -------
    WFPortfolioResult
        Validation result with verdict.
    """
    n_windows = len(selected_per_window)
    if n_windows == 0:
        return WFPortfolioResult(
            n_windows=0, selection_stability=0.0,
            core_strategies=[], peripheral_strategies=list(strategy_keys),
            oos_win_rate=0.0, oos_sharpes=[], oos_returns=[],
            mean_oos_sharpe=0.0, sharpe_consistency=0.0, verdict="FRAGILE",
        )

    # Selection stability: how often each strategy is picked
    selection_count: dict[str, int] = {}
    for sel in selected_per_window:
        for s in sel:
            selection_count[s] = selection_count.get(s, 0) + 1

    core = [s for s, c in selection_count.items() if c > n_windows * 0.5]
    peripheral = [s for s, c in selection_count.items() if c <= n_windows * 0.5]
    stability = len(core) / max(len(selection_count), 1)

    # OOS performance
    oos_win_rate = sum(1 for s in portfolio_oos_sharpes if s > 0) / max(n_windows, 1)
    mean_sharpe = float(np.mean(portfolio_oos_sharpes)) if portfolio_oos_sharpes else 0.0

    # Sharpe consistency: 1 - CV (coefficient of variation)
    if len(portfolio_oos_sharpes) > 1 and mean_sharpe != 0:
        cv = float(np.std(portfolio_oos_sharpes) / abs(mean_sharpe))
        consistency = max(0.0, 1.0 - cv)
    else:
        consistency = 0.0

    # Verdict
    if stability >= 0.6 and oos_win_rate >= 0.6 and mean_sharpe > 0:
        verdict = "ROBUST"
    elif stability >= 0.4 and oos_win_rate >= 0.4:
        verdict = "MARGINAL"
    else:
        verdict = "FRAGILE"

    return WFPortfolioResult(
        n_windows=n_windows,
        selection_stability=round(stability, 3),
        core_strategies=sorted(core),
        peripheral_strategies=sorted(peripheral),
        oos_win_rate=round(oos_win_rate, 3),
        oos_sharpes=portfolio_oos_sharpes,
        oos_returns=portfolio_oos_returns,
        mean_oos_sharpe=round(mean_sharpe, 3),
        sharpe_consistency=round(consistency, 3),
        verdict=verdict,
    )
