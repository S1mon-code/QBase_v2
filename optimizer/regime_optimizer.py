"""Orchestrator: optimize a strategy on regime-matched historical periods.

1. Load regime labels for instrument
2. Filter by regime + direction + split=train
3. Run Optuna two-phase optimization on those periods
4. Record all trials
5. Check robustness
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pipeline.qbase_config import ALPHAFORGE_PATH, PROJECT_ROOT

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from optimizer.config import COARSE_TRIALS, FINE_TRIALS, NARROW_RADIUS
from optimizer.core import BacktestMetrics, composite_objective
from optimizer.param_discovery import discover_params
from optimizer.robustness import check_robustness
from optimizer.trial_registry import TrialRegistry
from pipeline.backtest_runner import run_qbase_backtest
from regime.schema import load_labels



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
        label_path = PROJECT_ROOT / "data" / "regime_labels" / f"{instrument}.yaml"
        config = load_labels(label_path)

        periods: list[dict[str, str]] = []
        for lbl in config.labels:
            if lbl.regime != regime:
                continue
            if lbl.direction != direction:
                continue
            if lbl.split != "train":
                continue

            # Prefer buffer dates when available, else fall back to core dates
            start_date = lbl.buffer_start if lbl.buffer_start is not None else lbl.start
            end_date = lbl.buffer_end if lbl.buffer_end is not None else lbl.end

            periods.append({
                "start": str(start_date),
                "end": str(end_date),
            })

        return periods

    def _evaluate(
        self,
        strategy_class: type,
        params: dict[str, Any],
        periods: list[dict[str, str]],
        instrument: str,
        freq: str,
        baseline_sharpe: float,
        phase: str,
        signal_direction: str | None = None,
        industrial: bool = False,
    ) -> float:
        """Evaluate a parameter set on full continuous span of regime periods.

        Runs a single backtest spanning from earliest buffer_start to latest
        buffer_end. The strategy naturally produces no signal during non-regime
        segments (long-only clips to 0 during downtrends), so the equity curve
        stays flat in those gaps — matching real-world behavior.
        """
        if not periods:
            return -10.0

        # Find full span across all periods
        full_start = min(p["start"] for p in periods)
        full_end = max(p["end"] for p in periods)

        try:
            result = run_qbase_backtest(
                strategy_class,
                params,
                symbol=instrument,
                freq=freq,
                start=full_start,
                end=full_end,
                direction=signal_direction,
                active_periods=periods,
                industrial=industrial,
            )
        except Exception:
            return -10.0

        dr = result.daily_returns
        if hasattr(dr, "values"):
            dr = dr.values
        combined_returns = np.asarray(dr, dtype=np.float64)

        if len(combined_returns) == 0:
            return -10.0

        period_results = [result]

        # Aggregate scalar metrics
        sharpes = [r.sharpe for r in period_results]
        max_dds = [r.max_drawdown for r in period_results]
        n_trades_total = sum(r.n_trades for r in period_results)
        win_rates = [r.win_rate for r in period_results]

        # Use mean Sharpe across periods (weighted by period length)
        period_lengths = [len(combined_returns)]
        total_len = sum(period_lengths)
        if total_len == 0:
            return -10.0

        weights = [l / total_len for l in period_lengths]
        agg_sharpe = float(sum(s * w for s, w in zip(sharpes, weights)))
        agg_max_dd = float(max(max_dds))  # worst drawdown across periods
        agg_win_rate = float(sum(wr * w for wr, w in zip(win_rates, weights)))

        # Compute annualized return from combined returns
        if len(combined_returns) > 0:
            total_return = float(np.prod(1.0 + combined_returns) - 1.0)
            n_years = len(combined_returns) / 252.0
            if n_years > 0 and total_return > -1.0:
                agg_ann_return = float((1.0 + total_return) ** (1.0 / n_years) - 1.0)
            else:
                agg_ann_return = total_return
        else:
            agg_ann_return = 0.0

        # Compute profit factor from period results
        total_profit = sum(
            getattr(r, "total_profit", 0.0) for r in period_results
        )
        total_loss = sum(
            abs(getattr(r, "total_loss", 0.0)) for r in period_results
        )
        # Fallback: estimate from daily returns if period results lack profit/loss
        if total_profit == 0.0 and total_loss == 0.0:
            gains = combined_returns[combined_returns > 0]
            losses = combined_returns[combined_returns < 0]
            total_profit = float(np.sum(gains)) if len(gains) > 0 else 0.0
            total_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-10
        agg_pf = total_profit / total_loss if total_loss > 0 else 0.0

        # Compute max single trade contribution
        if hasattr(period_results[0], "trades") and period_results[0].trades is not None:
            all_trade_pnls = []
            for r in period_results:
                if hasattr(r, "trades") and r.trades is not None and len(r.trades) > 0:
                    pnls = r.trades.get("pnl", r.trades.get("profit", []))
                    if hasattr(pnls, "values"):
                        pnls = pnls.values
                    all_trade_pnls.extend(pnls)
            if all_trade_pnls:
                total_pnl = sum(abs(p) for p in all_trade_pnls)
                max_single = max(abs(p) for p in all_trade_pnls) / total_pnl if total_pnl > 0 else 0.0
            else:
                max_single = 0.0
        else:
            max_single = 0.0

        # Compute CVaR-95 from combined daily returns
        if len(combined_returns) > 0:
            pct_5 = np.percentile(combined_returns, 5)
            tail_returns = combined_returns[combined_returns <= pct_5]
            agg_cvar = float(np.mean(tail_returns)) if len(tail_returns) > 0 else float(pct_5)
        else:
            agg_cvar = 0.0

        # Compute skewness and kurtosis from combined returns
        if len(combined_returns) > 3:
            mean_r = np.mean(combined_returns)
            std_r = np.std(combined_returns)
            if std_r > 0:
                agg_skew = float(np.mean(((combined_returns - mean_r) / std_r) ** 3))
                agg_kurt = float(np.mean(((combined_returns - mean_r) / std_r) ** 4))
            else:
                agg_skew = 0.0
                agg_kurt = 3.0
        else:
            agg_skew = 0.0
            agg_kurt = 3.0

        # Build equity curve from combined returns
        equity_curve = np.cumprod(1.0 + combined_returns)

        metrics = BacktestMetrics(
            sharpe=agg_sharpe,
            annualized_return=agg_ann_return,
            max_drawdown=agg_max_dd,
            cvar_95=agg_cvar,
            n_trades=n_trades_total,
            win_rate=agg_win_rate,
            profit_factor=agg_pf,
            skewness=agg_skew,
            kurtosis=agg_kurt,
            daily_returns=combined_returns,
            equity_curve=equity_curve,
            max_single_trade_pct=max_single,
        )

        return composite_objective(metrics, baseline_sharpe, phase, freq)

    def _sample_params(
        self,
        trial: optuna.Trial,
        param_space: dict[str, dict],
    ) -> dict[str, Any]:
        """Sample parameters from param_space using the Optuna trial."""
        params: dict[str, Any] = {}
        for name, spec in param_space.items():
            ptype = spec.get("type", float)
            low = spec["low"]
            high = spec["high"]
            step = spec.get("step")

            if ptype is int or ptype == int:
                step_int = int(step) if step is not None else 1
                params[name] = trial.suggest_int(name, int(low), int(high), step=step_int)
            else:
                if step is not None:
                    params[name] = trial.suggest_float(name, float(low), float(high), step=float(step))
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high))
        return params

    def _run_coarse_phase(
        self,
        strategy_class: type,
        param_space: dict[str, dict],
        periods: list[dict[str, str]],
        instrument: str,
        freq: str,
        baseline_sharpe: float,
        n_trials: int = COARSE_TRIALS,
        signal_direction: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Phase 1: Coarse search with TPE over full parameter range.

        Uses tanh-compressed S_performance.
        """
        strategy_name = getattr(strategy_class, "name", strategy_class.__name__)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, param_space)
            score = self._evaluate(
                strategy_class, params, periods, instrument, freq, baseline_sharpe,
                phase="coarse", signal_direction=signal_direction,
            )

            # Record every trial to the registry
            # Use a placeholder sharpe; we don't have it separately so use score as proxy
            self._registry.record(
                strategy=strategy_name,
                params=params,
                sharpe=float(score),  # best proxy available without re-running
                score=float(score),
                regime=f"{instrument}_coarse",
                symbol=instrument,
                freq=freq,
                n_trades=0,
                status="coarse",
            )

            return score

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_score = study.best_value
        return best_params, best_score

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
        signal_direction: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Phase 2: Fine search around coarse best ±radius.

        Narrows the search space to ±radius around coarse_best for each param.
        Uses linear S_performance.
        """
        strategy_name = getattr(strategy_class, "name", strategy_class.__name__)

        # Build narrowed param space
        narrow_space: dict[str, dict] = {}
        for name, spec in param_space.items():
            if name not in coarse_best:
                narrow_space[name] = spec
                continue

            ptype = spec.get("type", float)
            center = coarse_best[name]
            orig_low = spec["low"]
            orig_high = spec["high"]
            step = spec.get("step")

            if isinstance(center, (int, float)) and center != 0:
                new_low = center * (1.0 - radius)
                new_high = center * (1.0 + radius)
            else:
                new_low = orig_low
                new_high = orig_high

            # Clamp to original bounds
            new_low = max(orig_low, new_low)
            new_high = min(orig_high, new_high)

            # Ensure low < high (edge case: very small range)
            if new_low >= new_high:
                new_low = orig_low
                new_high = orig_high

            if ptype is int or ptype == int:
                new_low = max(1, int(round(new_low)))
                new_high = max(new_low + 1, int(round(new_high)))

            narrow_space[name] = {
                "type": ptype,
                "default": spec.get("default", center),
                "low": new_low,
                "high": new_high,
                "step": step,
            }

        # Use industrial mode in fine phase for realistic cost estimation
        use_industrial = True

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=123),
        )

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, narrow_space)
            score = self._evaluate(
                strategy_class, params, periods, instrument, freq, baseline_sharpe,
                phase="fine", signal_direction=signal_direction,
                industrial=use_industrial,
            )

            self._registry.record(
                strategy=strategy_name,
                params=params,
                sharpe=float(score),
                score=float(score),
                regime=f"{instrument}_fine",
                symbol=instrument,
                freq=freq,
                n_trades=0,
                status="fine",
            )

            return score

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_score = study.best_value
        return best_params, best_score

    def optimize(
        self,
        strategy_class: type,
        instrument: str,
        freq: str,
        regime: str,
        direction: str,
        n_trials: int = 80,
        baseline_sharpe: float = 0.0,
        signal_direction: str | None = None,
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
        # Step 1: Load regime-matched training periods
        periods = self._load_regime_periods(instrument, regime, direction)
        if not periods:
            raise ValueError(
                f"No training periods found for instrument={instrument!r}, "
                f"regime={regime!r}, direction={direction!r}"
            )

        # Step 2: Discover optimizable parameters
        param_space = discover_params(strategy_class)

        # Determine trial split: coarse gets 30, fine gets 50 (ignore n_trials if using defaults)
        coarse_n = COARSE_TRIALS
        fine_n = FINE_TRIALS

        # Step 3: Coarse TPE search
        coarse_best, coarse_score = self._run_coarse_phase(
            strategy_class, param_space, periods, instrument, freq,
            baseline_sharpe, n_trials=coarse_n,
            signal_direction=signal_direction,
        )

        # Early exit if coarse phase is in the dead zone
        if coarse_score <= -10.0:
            return {
                "best_params": coarse_best,
                "best_score": coarse_score,
                "is_robust": False,
                "param_space": param_space,
                "n_periods": len(periods),
                "coarse_best": coarse_best,
                "coarse_score": coarse_score,
            }

        # Step 4: Fine search around coarse best
        fine_best, fine_score = self._run_fine_phase(
            strategy_class, coarse_best, param_space, periods, instrument, freq,
            baseline_sharpe, n_trials=fine_n, radius=NARROW_RADIUS,
            signal_direction=signal_direction,
        )

        # Step 5: Robustness check (also with industrial mode)
        def evaluate_fn(params: dict[str, Any]) -> float:
            return self._evaluate(
                strategy_class, params, periods, instrument, freq,
                baseline_sharpe, phase="fine",
                signal_direction=signal_direction,
                industrial=True,
            )

        robustness_result = check_robustness(
            best_params=fine_best,
            best_score=fine_score,
            evaluate_fn=evaluate_fn,
        )

        return {
            "best_params": fine_best,
            "best_score": fine_score,
            "is_robust": robustness_result["is_robust"],
            "param_space": param_space,
            "n_periods": len(periods),
            "coarse_best": coarse_best,
            "coarse_score": coarse_score,
        }
