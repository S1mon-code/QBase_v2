"""Central orchestration script for single-strategy development pipeline.

Runs baselines, bare logic gate, regime pre-validation, optimization,
6-layer validation, attribution, and report generation for a single strategy.

Usage:
    from pipeline.dev_pipeline import run_baselines, run_single_strategy_pipeline
    from strategies.trending.medium.v1 import TrendMediumV1

    baselines = run_baselines("I", "long", "strong_trend", freq="daily")
    result = run_single_strategy_pipeline(
        TrendMediumV1, "I", "long", "strong_trend", "medium", "v1"
    )
"""

from __future__ import annotations

import logging
import shutil
from typing import Any

import numpy as np

# ── Paths (from centralized config) ──────────────────────────────────────────
from pipeline.qbase_config import ALPHAFORGE_PATH, PROJECT_ROOT

# Suppress noisy loggers
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.getLogger("alphaforge").setLevel(logging.WARNING)

# ── AlphaForge imports ─────────────────────────────────────────────────────────
from alphaforge.report import HTMLReportGenerator

# ── QBase imports ──────────────────────────────────────────────────────────────
from pipeline.backtest_runner import run_qbase_backtest
from pipeline.utils import (
    _DIR_MAP,
    _HORIZON_BASELINE,
    _concat_daily_returns,
    _filter_labels,
    _generate_html_report,
    _get_trade_regimes,
    _load_bars_for_labels,
    _load_regime_labels,
    _profit_factor,
    _run_on_labels,
    _save_yaml,
    _total_bars,
    _total_n_trades,
    _weighted_mean_sharpe,
)
from strategies.baselines.tsmom_fast import TSMOMFast
from strategies.baselines.tsmom_medium import TSMOMMedium
from strategies.baselines.tsmom_slow import TSMOMSlow
from optimizer.regime_optimizer import RegimeOptimizer
from optimizer.trial_registry import TrialRegistry
from validation.pipeline import run_validation_pipeline
from validation.deflated_sharpe import deflated_sharpe_ratio
from validation.regime_cv import run_regime_cv
from validation.oos_validator import validate_oos
from validation.walk_forward import walk_forward_verdict
from validation.monte_carlo import bootstrap_test
from validation.permutation_test import permutation_test
from validation.industrial_check import check_industrial_decay
from validation.stress_test import run_stress_test
from attribution.horizon import horizon_attribution
from attribution.baseline import decompose_baseline
from attribution.regime import regime_attribution
from attribution.operational import operational_attribution
from attribution.report import generate_attribution_report


# ── Public API ─────────────────────────────────────────────────────────────────


def run_baselines(
    symbol: str,
    direction: str,
    regime: str,
    freq: str = "daily",
) -> dict:
    """Run TSMOM fast/medium/slow baselines on train regime periods.

    Filters labels by: split==train AND direction matches AND regime matches.
    Runs each baseline on each period, aggregates by period-length-weighted mean Sharpe.
    Saves HTML reports to research/{symbol}/{direction}/baselines/{horizon}.html.

    Args:
        symbol:    Instrument code, e.g. "I".
        direction: "long" or "short".
        regime:    Regime name, e.g. "strong_trend".
        freq:      Bar frequency.

    Returns:
        {"fast": sharpe, "medium": sharpe, "slow": sharpe}
    """
    print(f"\n  Running baselines: {symbol} {direction.upper()} | {regime} | {freq}")

    regime_config = _load_regime_labels(symbol)
    train_labels = _filter_labels(regime_config, "train", direction, regime)

    if not train_labels:
        print(f"  [baselines] No train periods found for {symbol}/{direction}/{regime}")
        return {"fast": 0.0, "medium": 0.0, "slow": 0.0}

    baselines_out = {}
    reporter = HTMLReportGenerator()

    for horizon, baseline_cls in _HORIZON_BASELINE.items():
        results = _run_on_labels(baseline_cls, {}, symbol, train_labels, freq)
        sharpe = _weighted_mean_sharpe(results)
        baselines_out[horizon] = sharpe

        # Save HTML report (use last result for single report, or best available)
        if results:
            report_dir = (
                PROJECT_ROOT / "research" / symbol / direction / "baselines"
            )
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{horizon}.html"
            try:
                # Use the result with most data for the report
                best_result = max(results, key=lambda r: len(
                    r.daily_returns.values if hasattr(r.daily_returns, "values") else r.daily_returns
                ))
                bar_data = _load_bars_for_labels(symbol, train_labels, freq)
                reporter.generate(best_result, str(report_path), bar_data=bar_data, freq=freq)
            except Exception as e:
                print(f"  [baselines] Report generation failed for {horizon}: {e}")

    print(
        f"  [baseline] fast: {baselines_out['fast']:.2f}, "
        f"medium: {baselines_out['medium']:.2f}, "
        f"slow: {baselines_out['slow']:.2f}"
    )
    return baselines_out


def run_single_strategy_pipeline(
    strategy_class: type,
    symbol: str,
    direction: str,
    regime: str,
    horizon: str,
    version: str,
    freq: str = "daily",
    params_override: dict | None = None,
) -> dict:
    """Full single-strategy development pipeline.

    Runs bare logic gate, vs-baseline check, regime pre-validation,
    optimization (or uses params_override), 6-layer validation,
    attribution, and report generation.

    Args:
        strategy_class:  QBase strategy class to develop.
        symbol:          Instrument code, e.g. "I".
        direction:       "long" or "short".
        regime:          Regime name, e.g. "strong_trend".
        horizon:         "fast", "medium", or "slow".
        version:         Version string, e.g. "v1".
        freq:            Bar frequency.
        params_override: If provided, skip optimization and use these params.

    Returns:
        Summary dict with all pipeline results.
    """
    strategy_name = getattr(strategy_class, "name", strategy_class.__name__)

    # ── Header ─────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  {symbol} {direction.upper()} │ {regime} │ {horizon} │ {version}")
    print(f"  strategy: {strategy_name}")
    print(f"{'═' * 60}\n")

    # Output directory — temp dir first, renamed with OOS return after report generation
    # Format: research/{regime}/{direction}/{instrument}/{timeframe}/v{N}_{+/-}{return}%
    if regime == "mean_reversion":
        _research_base = PROJECT_ROOT / "research" / regime / symbol / freq
    else:
        _research_base = PROJECT_ROOT / "research" / regime / direction / symbol / freq
    output_dir = _research_base / f"_{version}_temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load regime labels
    # OOS uses ALL direction-matching periods (not filtered by regime) so that
    # the OOS report reflects performance across all market conditions.
    regime_config = _load_regime_labels(symbol)
    train_labels = _filter_labels(regime_config, "train", direction, regime)
    oos_labels = _filter_labels(regime_config, "oos", direction)  # all regimes
    all_train_labels = _filter_labels(regime_config, "train", direction)

    if not train_labels:
        print(f"  [ERROR] No train periods found for {symbol}/{direction}/{regime}")
        return {
            "status": "no_train_periods",
            "symbol": symbol, "direction": direction, "regime": regime,
            "horizon": horizon, "version": version,
        }

    # ── Step B3 — Bare Logic Gate ───────────────────────────────────────────────
    print(f"{'─' * 40}")
    print("  B3: Bare Logic Gate")
    print(f"{'─' * 40}")

    # Run full continuous span for bare logic gate
    train_start = str(min(lbl.buffer_start or lbl.start for lbl in train_labels))
    train_end = str(max(lbl.buffer_end or lbl.end for lbl in train_labels))
    signal_direction = direction  # "long" or "short"

    # Build active_periods dicts for train and oos
    train_active = [{"start": str(lbl.start), "end": str(lbl.end)} for lbl in train_labels]
    oos_active = [{"start": str(lbl.start), "end": str(lbl.end)} for lbl in oos_labels] if oos_labels else []

    try:
        bare_full = run_qbase_backtest(
            strategy_class, {}, symbol, freq,
            start=train_start, end=train_end, direction=signal_direction,
            active_periods=train_active,
        )
        n_trades_bare = bare_full.n_trades
        bare_sharpe = bare_full.sharpe
        pf_bare = getattr(bare_full, "profit_factor", 1.0)
    except Exception as e:
        print(f"  [B3] Bare gate backtest error: {e}")
        n_trades_bare = 0
        bare_sharpe = 0.0
        pf_bare = 0.0

    # Gate thresholds
    min_trades = 30 if freq in ("1h", "4h") else 10

    gate_passed = n_trades_bare >= min_trades and pf_bare >= 1.0
    gate_status = "PASS" if gate_passed else "FAIL"
    print(f"  n_trades={n_trades_bare}  profit_factor={pf_bare:.2f}  sharpe={bare_sharpe:.2f}  [{gate_status}]")

    if not gate_passed:
        reasons = []
        if n_trades_bare < min_trades:
            reasons.append(f"n_trades={n_trades_bare} < {min_trades}")
        if pf_bare < 1.0:
            reasons.append(f"profit_factor={pf_bare:.2f} < 1.0")
        print(f"  [FAIL] Bare gate failed: {', '.join(reasons)}")
        return {
            "status": "failed_bare_gate",
            "symbol": symbol, "direction": direction, "regime": regime,
            "horizon": horizon, "version": version,
            "bare_gate": {
                "passed": False, "n_trades": n_trades_bare, "profit_factor": pf_bare,
                "reasons": reasons,
            },
        }

    # ── Step B4 — vs Baseline ──────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  B4: vs Baseline")
    print(f"{'─' * 40}")

    baselines = run_baselines(symbol, direction, regime, freq)
    baseline_sharpe = baselines.get(horizon, 0.0)
    delta = bare_sharpe - baseline_sharpe

    print(
        f"  [vs baseline] strategy={bare_sharpe:.2f}  "
        f"baseline={baseline_sharpe:.2f}  delta={delta:+.2f}"
    )
    if delta < 0:
        print(f"  [WARN] Strategy Sharpe below {horizon} baseline (soft warning)")

    # ── Step B5 — Regime Pre-validation ───────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  B5: Regime Pre-validation")
    print(f"{'─' * 40}")

    full_train_results = _run_on_labels(strategy_class, {}, symbol, all_train_labels, freq)
    full_train_sharpe = _weighted_mean_sharpe(full_train_results)

    print(
        f"  regime_sharpe={bare_sharpe:.2f}  full_history_sharpe={full_train_sharpe:.2f}"
    )
    if bare_sharpe <= full_train_sharpe:
        print("  [WARN] Regime-specific Sharpe not better than full history (soft warning)")
    else:
        print("  [OK] Signal fits regime")

    # ── Step C — Optimization ─────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  C: Optimization")
    print(f"{'─' * 40}")

    best_params: dict[str, Any]
    opt_score = 0.0
    is_robust = False
    n_opt_periods = len(train_labels)

    if params_override is not None:
        best_params = params_override
        print(f"  [optimize] Using params_override: {best_params}")
        opt_score = bare_sharpe
        is_robust = True
    else:
        try:
            registry = TrialRegistry()
            optimizer = RegimeOptimizer(registry=registry)
            opt_result = optimizer.optimize(
                strategy_class=strategy_class,
                instrument=symbol,
                freq=freq,
                regime=regime,
                direction=_DIR_MAP.get(direction, direction),
                baseline_sharpe=baseline_sharpe,
            )
            best_params = opt_result["best_params"]
            opt_score = opt_result["best_score"]
            is_robust = opt_result["is_robust"]
            n_opt_periods = opt_result["n_periods"]
            print(
                f"  [optimize] score={opt_score:.2f}  "
                f"robust={is_robust}  n_periods={n_opt_periods}"
            )
        except Exception as e:
            print(f"  [optimize] ERROR: {e} — falling back to default params")
            best_params = {}
            opt_score = bare_sharpe
            is_robust = False

    # Save params (complete format with metadata, ranges, indicators)
    from pipeline.report_files import generate_params_yaml, _save_yaml as _save_yaml_clean
    params_data = generate_params_yaml(
        strategy_class, best_params, opt_score, is_robust,
        symbol, freq, n_trials=0,  # updated in final companion file write
    )
    _save_yaml_clean(params_data, output_dir / "params.yaml")
    print(f"  [optimize] Params saved to {output_dir / 'params.yaml'}")

    # ── Step D — 6-Layer Validation ────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  D: 6-Layer Validation (LOCKED params)")
    print(f"{'─' * 40}")

    # D1: Regime CV (LOO) — full span continuous, mask out one period at a time
    fold_sharpes: list[float] = []
    for i, held_out in enumerate(train_labels):
        # Build active_periods excluding the held-out period
        loo_active = [
            {"start": str(lbl.start), "end": str(lbl.end)}
            for j, lbl in enumerate(train_labels) if j != i
        ]
        try:
            r = run_qbase_backtest(
                strategy_class, best_params, symbol, freq,
                start=train_start, end=train_end,
                direction=signal_direction,
                active_periods=loo_active,
            )
            fold_sharpes.append(r.sharpe)
        except Exception as e:
            print(f"  [D1] fold {i} skip: {e}")

    cv_result = None
    if fold_sharpes:
        try:
            cv_result = run_regime_cv(fold_sharpes, strategy_name, regime)
            print(f"  [D1 regime_cv] verdict={cv_result.verdict}  mean={cv_result.mean_sharpe:.2f}")
        except Exception as e:
            print(f"  [D1] CV error: {e}")

    # D2: OOS — full continuous span
    oos_full_result = None
    oos_sharpe = 0.0
    oos_n_trades = 0
    oos_daily_returns = np.array([], dtype=np.float64)
    oos_results = []  # keep for compatibility with downstream code

    if oos_labels:
        oos_start = str(min(lbl.buffer_start or lbl.start for lbl in oos_labels))
        oos_end = str(max(lbl.buffer_end or lbl.end for lbl in oos_labels))
        try:
            oos_full_result = run_qbase_backtest(
                strategy_class, best_params, symbol, freq,
                start=oos_start, end=oos_end, direction=signal_direction,
                active_periods=oos_active,
            )
            oos_sharpe = oos_full_result.sharpe
            oos_n_trades = oos_full_result.n_trades
            dr = oos_full_result.daily_returns
            if hasattr(dr, "values"):
                dr = dr.values
            oos_daily_returns = np.asarray(dr, dtype=np.float64)
            oos_results = [oos_full_result]
        except Exception as e:
            print(f"  [D2] OOS backtest error: {e}")

    # IS — full continuous span
    is_full_result = None
    is_sharpe = 0.0
    is_n_trades = 0
    is_results_locked = []

    try:
        is_full_result = run_qbase_backtest(
            strategy_class, best_params, symbol, freq,
            start=train_start, end=train_end, direction=signal_direction,
            active_periods=train_active,
        )
        is_sharpe = is_full_result.sharpe
        is_n_trades = is_full_result.n_trades
        is_results_locked = [is_full_result]
    except Exception as e:
        print(f"  [D2] IS backtest error: {e}")

    # Average hold time from trades
    def _avg_hold(results_list) -> float:
        total_hold = 0.0
        total_t = 0
        for r in results_list:
            try:
                trades = r.trades
                if trades is None or len(trades) == 0:
                    continue
                if "hold_bars" in trades.columns:
                    total_hold += trades["hold_bars"].sum()
                    total_t += len(trades)
            except Exception:
                continue
        return total_hold / total_t if total_t > 0 else 0.0

    is_avg_hold = _avg_hold(is_results_locked)
    oos_avg_hold = _avg_hold(oos_results)

    oos_result_obj = None
    if oos_labels:
        try:
            oos_result_obj = validate_oos(
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                is_trades=is_n_trades,
                oos_trades=oos_n_trades,
                is_avg_hold=is_avg_hold,
                oos_avg_hold=oos_avg_hold,
            )
            print(f"  [D2 oos] is={is_sharpe:.2f}  oos={oos_sharpe:.2f}  flags={list(oos_result_obj.flags)}")
        except Exception as e:
            print(f"  [D2] OOS error: {e}")
    else:
        print("  [D2] No OOS labels — skipping OOS validation")

    # D3: Walk-Forward (LOO on train folds with fixed params)
    window_sharpes = fold_sharpes  # same fold sharpes as D1
    wf_result_obj = None
    if window_sharpes:
        try:
            wf_result_obj = walk_forward_verdict(window_sharpes, "regime_aware")
            print(f"  [D3 walk_forward] passed={wf_result_obj.passed}  mean={wf_result_obj.mean_sharpe:.2f}")
        except Exception as e:
            print(f"  [D3] WF error: {e}")

    # D4: Deflated Sharpe
    dsr_value = None
    sharpe_std = float(np.std(fold_sharpes)) if len(fold_sharpes) > 1 else 0.0
    n_obs = _total_bars(is_results_locked)
    n_obs = max(n_obs, 1)

    try:
        registry = TrialRegistry()
        n_trials_recorded = len(registry.get_trials_for_strategy(strategy_name))
        n_trials_for_dsr = max(n_trials_recorded, 1)
    except Exception:
        n_trials_for_dsr = 1

    if fold_sharpes and len(oos_daily_returns) > 0:
        try:
            dsr_value = deflated_sharpe_ratio(
                oos_sharpe, n_trials_for_dsr, sharpe_std, n_obs
            )
            print(f"  [D4 deflated_sharpe] dsr={dsr_value:.3f}  n_trials={n_trials_for_dsr}")
        except Exception as e:
            print(f"  [D4] DSR error: {e}")

    # D5a: Bootstrap
    bootstrap_result_obj = None
    if len(oos_daily_returns) >= 30:
        try:
            bootstrap_result_obj = bootstrap_test(oos_daily_returns)
            print(f"  [D5a bootstrap] verdict={bootstrap_result_obj.verdict}  CI=[{bootstrap_result_obj.sharpe_ci_lower:.3f}, {bootstrap_result_obj.sharpe_ci_upper:.3f}]")
        except Exception as e:
            print(f"  [D5a] Bootstrap error: {e}")

    # D5b: Permutation
    perm_result_obj = None
    if len(oos_daily_returns) >= 30 and oos_sharpe != 0.0:
        try:
            perm_result_obj = permutation_test(oos_daily_returns, oos_sharpe)
            print(f"  [D5b permutation] verdict={perm_result_obj.verdict}  p_value={perm_result_obj.p_value:.3f}")
        except Exception as e:
            print(f"  [D5b] Permutation error: {e}")

    # D6a: Industrial
    industrial_oos_results = _run_on_labels(
        strategy_class, best_params, symbol, oos_labels, freq, industrial=True
    )
    industrial_oos_sharpe = _weighted_mean_sharpe(industrial_oos_results)

    industrial_result_obj = None
    if oos_labels:
        try:
            industrial_result_obj = check_industrial_decay(oos_sharpe, industrial_oos_sharpe)
            print(
                f"  [D6a industrial] basic={oos_sharpe:.2f}  "
                f"industrial={industrial_oos_sharpe:.2f}  "
                f"verdict={industrial_result_obj.verdict}"
            )
        except Exception as e:
            print(f"  [D6a] Industrial error: {e}")

    # D6b: Stress (doubled slippage on OOS)
    stress_results = _run_on_labels(
        strategy_class, best_params, symbol, oos_labels, freq,
        config_overrides={"slippage_ticks": 2.0},
    )
    doubled_slippage_sharpe = _weighted_mean_sharpe(stress_results)

    stress_result_obj = None
    if oos_labels:
        try:
            stress_result_obj = run_stress_test(oos_sharpe, doubled_slippage_sharpe)
            print(
                f"  [D6b stress] oos={oos_sharpe:.2f}  "
                f"doubled_slip={doubled_slippage_sharpe:.2f}  "
                f"sensitivity={stress_result_obj.slippage_sensitivity}"
            )
        except Exception as e:
            print(f"  [D6b] Stress error: {e}")

    # Run validation pipeline
    val_result = None
    try:
        # Compute skew/kurt from OOS returns for DSR
        oos_skew = 0.0
        oos_kurt = 3.0
        if len(oos_daily_returns) > 3:
            m = np.mean(oos_daily_returns)
            s = np.std(oos_daily_returns)
            if s > 0:
                oos_skew = float(np.mean(((oos_daily_returns - m) / s) ** 3))
                oos_kurt = float(np.mean(((oos_daily_returns - m) / s) ** 4))

        val_result = run_validation_pipeline(
            fold_sharpes=fold_sharpes if fold_sharpes else None,
            strategy=strategy_name,
            regime=regime,
            is_sharpe=is_sharpe if oos_labels else None,
            oos_sharpe=oos_sharpe if oos_labels else None,
            is_trades=is_n_trades,
            oos_trades=oos_n_trades,
            is_avg_hold=is_avg_hold,
            oos_avg_hold=oos_avg_hold,
            window_sharpes=window_sharpes if window_sharpes else None,
            wf_mode="regime_aware",
            observed_sharpe=oos_sharpe if oos_labels else None,
            n_trials=n_trials_for_dsr,
            sharpe_std=sharpe_std if fold_sharpes else None,
            n_obs=n_obs if fold_sharpes else None,
            skew=oos_skew,
            kurt=oos_kurt,
            daily_returns=oos_daily_returns if len(oos_daily_returns) >= 30 else None,
            basic_sharpe=oos_sharpe if oos_labels else None,
            industrial_sharpe=industrial_oos_sharpe if oos_labels else None,
            doubled_slippage_sharpe=doubled_slippage_sharpe if oos_labels else None,
        )
        print(f"\n  [validation] hard_reject={val_result.hard_reject}")
        if val_result.reject_reasons:
            print(f"  [validation] reject_reasons={list(val_result.reject_reasons)}")
        if val_result.soft_flags:
            print(f"  [validation] soft_flags={list(val_result.soft_flags)}")
    except Exception as e:
        print(f"  [validation] Pipeline error: {e}")

    # Save validation results — extract all metrics needed by portfolio selection
    oos_total_return = 0.0
    oos_ann_return = 0.0
    oos_max_dd = 0.0
    oos_pf = 0.0
    oos_max_single_trade = 0.0

    if oos_full_result is not None:
        oos_total_return = float(getattr(oos_full_result, "total_return", 0.0))
        oos_ann_return = float(getattr(oos_full_result, "annualized_return", 0.0))
        oos_max_dd = float(getattr(oos_full_result, "max_drawdown", 0.0))
        oos_pf = float(getattr(oos_full_result, "profit_factor", 0.0))
        # Compute max single trade contribution
        try:
            trades = oos_full_result.trades
            if trades is not None and len(trades) > 0:
                pnl_col = "net_pnl" if "net_pnl" in trades.columns else "profit"
                if pnl_col in trades.columns:
                    pnls = trades[pnl_col].values
                    total_abs = sum(abs(p) for p in pnls)
                    if total_abs > 0:
                        oos_max_single_trade = float(max(abs(p) for p in pnls) / total_abs)
        except Exception:
            pass

    # Per-period return breakdown
    oos_period_returns = []
    if oos_labels and oos_full_result is not None:
        for lbl in oos_labels:
            try:
                pr = run_qbase_backtest(
                    strategy_class, best_params, symbol, freq,
                    start=str(lbl.start), end=str(lbl.end), direction=signal_direction,
                )
                oos_period_returns.append({
                    "period": f"{lbl.start} → {lbl.end}",
                    "total_return": float(getattr(pr, "total_return", 0.0)),
                    "sharpe": float(pr.sharpe),
                    "n_trades": int(pr.n_trades),
                })
            except Exception:
                pass

    # Check per-period concentration
    period_concentration_warning = False
    if len(oos_period_returns) >= 2:
        period_rets = [p["total_return"] for p in oos_period_returns]
        total_abs_ret = sum(abs(r) for r in period_rets)
        if total_abs_ret > 0:
            max_contribution = max(abs(r) for r in period_rets) / total_abs_ret
            if max_contribution > 0.70:
                period_concentration_warning = True

    from pipeline.report_files import generate_validation_yaml, _save_yaml as _save_yaml_clean

    oos_full_span_dict = {
        "start": str(min(lbl.buffer_start or lbl.start for lbl in oos_labels)) if oos_labels else None,
        "end": str(max(lbl.buffer_end or lbl.end for lbl in oos_labels)) if oos_labels else None,
        "total_return": oos_total_return,
        "annualized_return": oos_ann_return,
        "sharpe": float(oos_sharpe),
        "n_trades": int(oos_n_trades),
        "max_drawdown": oos_max_dd,
        "profit_factor": oos_pf,
        "max_single_trade_pct": oos_max_single_trade,
    }
    val_data = generate_validation_yaml(
        cv_result=cv_result,
        oos_result=oos_result_obj,
        oos_full_span=oos_full_span_dict,
        oos_period_breakdown=oos_period_returns,
        period_concentration_warning=period_concentration_warning,
        wf_result=wf_result_obj,
        dsr_value=dsr_value,
        observed_sharpe=float(oos_sharpe) if oos_labels else None,
        n_trials=n_trials_for_dsr,
        sharpe_std=sharpe_std if fold_sharpes else None,
        n_obs=n_obs if fold_sharpes else None,
        bootstrap_result=bootstrap_result_obj,
        perm_result=perm_result_obj,
        industrial_result=industrial_result_obj,
        stress_result=stress_result_obj,
        val_pipeline_result=val_result,
    )
    _save_yaml_clean(val_data, output_dir / "validation.yaml")

    # ── Step D Reports ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  D Reports")
    print(f"{'─' * 40}")

    # Train report — use already-computed full span IS result
    if is_full_result is not None:
        try:
            train_bar_data = _load_bars_for_labels(symbol, train_labels, freq)
            _generate_html_report(is_full_result, output_dir / "train.html", bar_data=train_bar_data, freq=freq)
            print(f"  [report] train.html saved (full span: {train_start} → {train_end})")
        except Exception as e:
            print(f"  [report] train.html failed: {e}")

    # OOS report — use already-computed full span OOS result
    if oos_full_result is not None:
        try:
            oos_bar_data = _load_bars_for_labels(symbol, oos_labels, freq)
            _generate_html_report(oos_full_result, output_dir / "oos.html", bar_data=oos_bar_data, freq=freq)
            oos_start_str = str(min(lbl.buffer_start or lbl.start for lbl in oos_labels))
            oos_end_str = str(max(lbl.buffer_end or lbl.end for lbl in oos_labels))
            print(f"  [report] oos.html saved (full span: {oos_start_str} → {oos_end_str})")
        except Exception as e:
            print(f"  [report] oos.html failed: {e}")

    # ── Step E — Attribution ───────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  E: Attribution")
    print(f"{'─' * 40}")

    horizon_result = None
    regime_attr_result = None
    baseline_decomp_result = None
    operational_result = None

    if len(oos_daily_returns) >= 10:

        # E-B: Horizon attribution
        try:
            fast_oos = _run_on_labels(TSMOMFast, {}, symbol, oos_labels, freq)
            medium_oos = _run_on_labels(TSMOMMedium, {}, symbol, oos_labels, freq)
            slow_oos = _run_on_labels(TSMOMSlow, {}, symbol, oos_labels, freq)

            tsmom_1m = _concat_daily_returns(fast_oos)
            tsmom_3m = _concat_daily_returns(medium_oos)
            tsmom_12m = _concat_daily_returns(slow_oos)

            # Align lengths
            min_len = min(len(oos_daily_returns), len(tsmom_1m), len(tsmom_3m), len(tsmom_12m))
            if min_len >= 10:
                horizon_result = horizon_attribution(
                    oos_daily_returns[:min_len],
                    tsmom_1m[:min_len],
                    tsmom_3m[:min_len],
                    tsmom_12m[:min_len],
                )
                fp = horizon_result.horizon_fingerprint
                print(
                    f"  [E-B horizon] fast={fp.get('fast', 0):.1f}%  "
                    f"medium={fp.get('medium', 0):.1f}%  "
                    f"slow={fp.get('slow', 0):.1f}%  "
                    f"alpha={horizon_result.independent_alpha:.4f}"
                )
        except Exception as e:
            print(f"  [E-B] Horizon attribution error: {e}")

        # E-C: Regime attribution
        try:
            # Collect all OOS results and their regime labels
            all_oos_trade_pnls = np.array([])
            all_oos_trade_regimes = np.array([])

            for r in oos_results:
                pnls, tags = _get_trade_regimes(r, oos_labels)
                if len(pnls) > 0:
                    all_oos_trade_pnls = np.concatenate([all_oos_trade_pnls, pnls])
                    all_oos_trade_regimes = np.concatenate([all_oos_trade_regimes, tags])

            if len(all_oos_trade_pnls) >= 5:
                regime_attr_result = regime_attribution(all_oos_trade_pnls, all_oos_trade_regimes)
                print(
                    f"  [E-C regime] best={regime_attr_result.best_regime}  "
                    f"worst={regime_attr_result.worst_regime}  "
                    f"dependent={regime_attr_result.regime_dependent}"
                )
        except Exception as e:
            print(f"  [E-C] Regime attribution error: {e}")

        # E-D: Baseline decomposition
        try:
            horizon_cls = _HORIZON_BASELINE.get(horizon, TSMOMMedium)
            bl_oos = _run_on_labels(horizon_cls, {}, symbol, oos_labels, freq)
            bl_returns = _concat_daily_returns(bl_oos)
            min_len = min(len(oos_daily_returns), len(bl_returns))
            if min_len >= 10:
                baseline_decomp_result = decompose_baseline(
                    oos_daily_returns[:min_len],
                    bl_returns[:min_len],
                )
                print(
                    f"  [E-D baseline] tsmom={baseline_decomp_result.tsmom_pct:.1f}%  "
                    f"alpha={baseline_decomp_result.alpha_pct:.1f}%  "
                    f"R2={baseline_decomp_result.r_squared:.3f}"
                )
        except Exception as e:
            print(f"  [E-D] Baseline decomposition error: {e}")

        # E-E: Operational
        try:
            operational_result = operational_attribution(
                basic_sharpe=float(oos_sharpe),
                industrial_sharpe=float(industrial_oos_sharpe),
            )
            decay_pct = (
                operational_result.total_decay / operational_result.basic_sharpe * 100.0
                if operational_result.basic_sharpe != 0.0
                else 0.0
            )
            print(f"  [E-E operational] decay={decay_pct:.1f}%  basic={oos_sharpe:.2f}  industrial={industrial_oos_sharpe:.2f}")
        except Exception as e:
            print(f"  [E-E] Operational attribution error: {e}")

        # Generate attribution report (all 5 layers)
        try:
            from pipeline.report_files import generate_attribution_md
            attr_md = generate_attribution_md(
                signal_result=None,  # TODO: add Layer A signal attribution
                horizon_result=horizon_result,
                regime_result=regime_attr_result,
                baseline_result=baseline_decomp_result,
                operational_result=operational_result,
                strategy_name=strategy_name,
                symbol=symbol,
            )
            attr_path = output_dir / "attribution.md"
            attr_path.write_text(attr_md, encoding="utf-8")
            print(f"  [attribution] Saved to {attr_path}")
        except Exception as e:
            print(f"  [attribution] Report generation error: {e}")
    else:
        print("  [E] Not enough OOS returns for attribution (need >= 10 bars)")

    # ── Final params.yaml rewrite with n_trials ──────────────────────────────
    try:
        params_final = generate_params_yaml(
            strategy_class, best_params, opt_score, is_robust,
            symbol, freq, n_trials=n_trials_for_dsr,
        )
        _save_yaml_clean(params_final, output_dir / "params.yaml")
    except Exception:
        pass  # early write already exists

    # ── Rename folder with OOS Total Return ──────────────────────────────────
    # Use result.total_return directly instead of parsing HTML regex
    if oos_full_result is not None:
        ret_pct = oos_full_result.total_return * 100
        sign = "+" if ret_pct >= 0 else ""
        final_name = f"{version}_{sign}{ret_pct:.2f}%"
    else:
        final_name = version

    final_dir = _research_base / final_name
    if final_dir.exists() and final_dir != output_dir:
        shutil.rmtree(final_dir)
    if output_dir != final_dir:
        output_dir.rename(final_dir)
        output_dir = final_dir
    print(f"  [rename] {final_name}")

    # Clean up old folders for this version (different return values)
    for old_dir in _research_base.iterdir():
        if (old_dir.is_dir()
            and old_dir.name.startswith(f"{version}_")
            and old_dir != output_dir):
            shutil.rmtree(old_dir)
            print(f"  [cleanup] removed {old_dir.name}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  PIPELINE COMPLETE: {symbol} {direction.upper()} | {regime} | {horizon} | {version}")
    print(f"  is_sharpe={is_sharpe:.2f}  oos_sharpe={oos_sharpe:.2f}  industrial={industrial_oos_sharpe:.2f}")
    hard_reject = val_result.hard_reject if val_result else False
    print(f"  hard_reject={hard_reject}")
    print(f"  output_dir={output_dir}")
    print(f"{'═' * 60}\n")

    return {
        "status": "completed",
        "symbol": symbol,
        "direction": direction,
        "regime": regime,
        "horizon": horizon,
        "version": version,
        "best_params": best_params,
        "baselines": baselines,
        "bare_gate": {
            "passed": gate_passed,
            "n_trades": n_trades_bare,
            "profit_factor": pf_bare,
        },
        "optimization": {
            "score": float(opt_score),
            "is_robust": is_robust,
        },
        "validation": {
            "hard_reject": val_result.hard_reject if val_result else False,
            "reject_reasons": list(val_result.reject_reasons) if val_result else [],
            "soft_flags": list(val_result.soft_flags) if val_result else [],
            "oos_sharpe": float(oos_sharpe),
            "industrial_oos_sharpe": float(industrial_oos_sharpe),
        },
        "output_dir": str(output_dir),
        "folder_name": final_name,
    }
