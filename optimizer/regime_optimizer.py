"""Orchestrator: optimize a strategy on regime-matched historical periods.

1. Load regime labels for instrument
2. Filter by regime + direction + split=train
3. Run Optuna two-phase optimization on those periods
4. Record all trials
5. Check robustness
"""

from __future__ import annotations

from typing import Any

from optimizer.config import COARSE_TRIALS, FINE_TRIALS, NARROW_RADIUS
from optimizer.core import BacktestMetrics, composite_objective
from optimizer.param_discovery import discover_params
from optimizer.robustness import check_robustness, multi_seed_optimize
from optimizer.trial_registry import TrialRegistry


# NOTE: See AlphaForge CLAUDE.md for backtesting API
def run_backtest(
    strategy: Any,
    symbol: str,
    freq: str,
    start: str,
    end: str,
    config: dict[str, Any] | None = None,
) -> BacktestMetrics:
    """Placeholder - will be connected to AlphaForge V6.0 engine."""
    raise NotImplementedError("Connect to AlphaForge V6.0")


class RegimeOptimizer:
    """Optimize a strategy on regime-matched historical periods.

    Pipeline:
    1. Load regime labels for instrument from data/regime_labels/{symbol}.yaml
    2. Filter by regime + direction + split=train
    3. Auto-discover parameters from strategy class
    4. Run Optuna two-phase optimization (coarse → fine)
    5. Record all trials to TrialRegistry
    6. Check robustness (plateau detection)
    """

    def __init__(self, registry: TrialRegistry | None = None) -> None:
        self._registry = registry or TrialRegistry()

    @property
    def registry(self) -> TrialRegistry:
        return self._registry

    def _load_regime_periods(
        self,
        instrument: str,
        regime: str,
        direction: str,
    ) -> list[dict[str, str]]:
        """Load and filter regime periods for training.

        Returns list of {"start": ..., "end": ...} dicts for train split.
        """
        # NOTE: See AlphaForge CLAUDE.md for regime label loading
        raise NotImplementedError("Connect to regime label loader")

    def _evaluate(
        self,
        strategy_class: type,
        params: dict[str, Any],
        periods: list[dict[str, str]],
        instrument: str,
        freq: str,
        baseline_sharpe: float,
        phase: str,
    ) -> float:
        """Evaluate a parameter set across all regime periods.

        Instantiates strategy with params, runs backtest on each period,
        aggregates metrics, and computes composite objective.
        """
        # NOTE: See AlphaForge CLAUDE.md for backtesting setup
        raise NotImplementedError("Connect to AlphaForge V6.0 for actual backtesting")

    def _run_coarse_phase(
        self,
        strategy_class: type,
        param_space: dict[str, dict],
        periods: list[dict[str, str]],
        instrument: str,
        freq: str,
        baseline_sharpe: float,
        n_trials: int = COARSE_TRIALS,
    ) -> tuple[dict[str, Any], float]:
        """Phase 1: Coarse search with TPE over full parameter range.

        Uses tanh-compressed S_performance.
        Includes probe trials for early stopping.
        """
        # NOTE: See AlphaForge CLAUDE.md for backtesting setup
        raise NotImplementedError("Connect to AlphaForge V6.0 for actual backtesting")

    def _run_fine_phase(
        self,
        strategy_class: type,
        coarse_best: dict[str, Any],
        param_space: dict[str, dict],
        periods: list[dict[str, str]],
        instrument: str,
        freq: str,
        baseline_sharpe: float,
        n_trials: int = FINE_TRIALS,
        radius: float = NARROW_RADIUS,
    ) -> tuple[dict[str, Any], float]:
        """Phase 2: Fine search around coarse best ±radius.

        Uses linear S_performance.
        1h+ strategies must use Industrial backtest mode.
        """
        # NOTE: See AlphaForge CLAUDE.md for backtesting setup
        raise NotImplementedError("Connect to AlphaForge V6.0 for actual backtesting")

    def optimize(
        self,
        strategy_class: type,
        instrument: str,
        freq: str,
        regime: str,
        direction: str,
        n_trials: int = 80,
        baseline_sharpe: float = 0.0,
    ) -> dict[str, Any]:
        """Full optimization pipeline.

        Steps:
        1. Load regime-matched training periods
        2. Discover optimizable parameters
        3. Phase 1: Coarse TPE search (30 trials)
        4. Phase 2: Fine search around best (50 trials)
        5. Robustness check (plateau detection)
        6. Record all trials

        Returns:
            {
                "best_params": dict,
                "best_score": float,
                "is_robust": bool,
                "param_space": dict,
                "n_periods": int,
                "coarse_best": dict,
                "coarse_score": float,
            }
        """
        # NOTE: See AlphaForge CLAUDE.md for backtesting setup
        raise NotImplementedError("Connect to AlphaForge V6.0 for actual backtesting")
