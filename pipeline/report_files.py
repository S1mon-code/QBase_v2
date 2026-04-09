"""Unified generator for strategy companion files.

Every strategy report (oos.html) must be accompanied by:
    - params.yaml   — strategy identity, parameters, indicators
    - attribution.md — 5-layer alpha source analysis
    - validation.yaml — 8-sublayer robustness verification

Usage:
    from pipeline.report_files import generate_companion_files

    generate_companion_files(
        output_dir=Path("research/long/long/I/4h/v27_+44.71%"),
        strategy_class=MildTrendLongI4hV27,
        best_params={"mcginley_period": 20},
        opt_score=0.79, is_robust=True,
        symbol="I", freq="4h",
        # validation inputs ...
        # attribution inputs ...
    )
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from attribution.baseline import BaselineDecomposition
from attribution.horizon import HorizonAttributionResult
from attribution.operational import OperationalAttribution
from attribution.regime import RegimeAttributionResult
from attribution.report import generate_attribution_report
from attribution.signal import SignalAttributionResult
from validation.industrial_check import IndustrialResult
from validation.monte_carlo import BootstrapResult
from validation.oos_validator import OOSResult
from validation.permutation_test import PermutationResult
from validation.pipeline import ValidationPipelineResult
from validation.regime_cv import RegimeCVResult
from validation.stress_test import StressTestResult
from validation.walk_forward import WalkForwardResult


# ── YAML helpers ─────────────────────────────────────────────────────────────

def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy/dataclass types to native Python for YAML."""
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _sanitize(dataclasses.asdict(obj))
    return obj


def _save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            _sanitize(data), f,
            default_flow_style=False, allow_unicode=True, sort_keys=False,
        )


# ── 1. params.yaml ──────────────────────────────────────────────────────────

def generate_params_yaml(
    strategy_class: type,
    best_params: dict[str, Any],
    opt_score: float,
    is_robust: bool,
    symbol: str,
    freq: str,
    n_trials: int = 0,
) -> dict:
    """Generate complete params.yaml content.

    Includes strategy metadata, actual parameter values with ranges,
    and indicator list.
    """
    from optimizer.param_discovery import discover_params

    # Instantiate strategy with best params applied
    strategy = strategy_class()
    for k, v in best_params.items():
        setattr(strategy, k, v)

    # Discover parameter ranges
    try:
        discovered = discover_params(strategy_class)
    except Exception:
        discovered = {}

    # Build parameters section with actual values + ranges
    parameters: dict[str, Any] = {}
    for name, info in discovered.items():
        parameters[name] = {
            "value": best_params.get(name, info.get("default")),
            "type": info.get("type", type(info.get("default", 0))).__name__,
            "range": [info.get("low"), info.get("high")],
            "step": info.get("step"),
        }
    # Include params that weren't discovered but were optimized
    for name, val in best_params.items():
        if name not in parameters:
            parameters[name] = {"value": val, "type": type(val).__name__}

    # Get indicator config
    indicators = []
    try:
        # Need arrays for get_indicator_config to work — use empty arrays
        # Just get the config metadata, not actual array data
        raw_configs = strategy.get_indicator_config()
        for ic in raw_configs:
            entry: dict[str, Any] = {"name": ic.get("name", "unknown")}
            if "params" in ic:
                entry["params"] = ic["params"]
            indicators.append(entry)
    except Exception:
        pass

    return {
        "strategy": {
            "name": getattr(strategy_class, "name", strategy_class.__name__),
            "class": strategy_class.__name__,
            "regime": getattr(strategy_class, "regime", ""),
            "horizon": getattr(strategy_class, "horizon", None),
            "direction": getattr(strategy_class, "direction", ""),
            "instrument": symbol,
            "timeframe": freq,
            "signal_dimensions": list(
                getattr(strategy_class, "signal_dimensions", [])
            ),
            "warmup": getattr(strategy_class, "warmup", 0),
        },
        "parameters": parameters,
        "optimization": {
            "score": float(opt_score),
            "is_robust": bool(is_robust),
            "n_trials": int(n_trials),
        },
        "indicators": indicators,
    }


# ── 2. validation.yaml ──────────────────────────────────────────────────────

def generate_validation_yaml(
    *,
    # Layer 1: Regime CV
    cv_result: RegimeCVResult | None = None,
    # Layer 2: OOS
    oos_result: OOSResult | None = None,
    oos_full_span: dict | None = None,
    oos_period_breakdown: list[dict] | None = None,
    period_concentration_warning: bool = False,
    # Layer 3: Walk-Forward
    wf_result: WalkForwardResult | None = None,
    # Layer 4: Deflated Sharpe
    dsr_value: float | None = None,
    observed_sharpe: float | None = None,
    n_trials: int | None = None,
    sharpe_std: float | None = None,
    n_obs: int | None = None,
    # Layer 5a: Bootstrap
    bootstrap_result: BootstrapResult | None = None,
    # Layer 5b: Permutation
    perm_result: PermutationResult | None = None,
    # Layer 6a: Industrial
    industrial_result: IndustrialResult | None = None,
    # Layer 6b: Stress
    stress_result: StressTestResult | None = None,
    # Aggregated verdict
    val_pipeline_result: ValidationPipelineResult | None = None,
) -> dict:
    """Generate complete validation.yaml with all 8 sub-layers."""

    data: dict[str, Any] = {}

    # Layer 1: Regime CV
    if cv_result is not None:
        data["regime_cv"] = {
            "verdict": cv_result.verdict,
            "n_folds": cv_result.n_folds,
            "fold_sharpes": list(cv_result.fold_sharpes),
            "mean_sharpe": cv_result.mean_sharpe,
            "std_sharpe": cv_result.std_sharpe,
            "win_rate": cv_result.win_rate,
        }
    else:
        data["regime_cv"] = None

    # Layer 2: OOS
    oos_section: dict[str, Any] = {}
    if oos_result is not None:
        oos_section["is_sharpe"] = oos_result.is_sharpe
        oos_section["oos_sharpe"] = oos_result.oos_sharpe
        oos_section["wf_ratio"] = oos_result.wf_ratio
        oos_section["industrial_sharpe"] = oos_result.industrial_sharpe
        oos_section["industrial_decay"] = oos_result.industrial_decay
        oos_section["flags"] = list(oos_result.flags)
    if oos_full_span is not None:
        oos_section["full_span"] = oos_full_span
    if oos_period_breakdown is not None:
        oos_section["period_breakdown"] = oos_period_breakdown
    oos_section["period_concentration_warning"] = period_concentration_warning
    data["oos"] = oos_section if oos_section else None

    # Layer 3: Walk-Forward
    if wf_result is not None:
        data["walk_forward"] = {
            "mode": wf_result.mode,
            "n_windows": wf_result.n_windows,
            "window_sharpes": list(wf_result.window_sharpes),
            "mean_sharpe": wf_result.mean_sharpe,
            "win_rate": wf_result.win_rate,
            "worst_sharpe": wf_result.worst_sharpe,
            "best_sharpe": wf_result.best_sharpe,
            "passed": wf_result.passed,
        }
    else:
        data["walk_forward"] = None

    # Layer 4: Deflated Sharpe
    data["deflated_sharpe"] = {
        "dsr": dsr_value,
        "observed_sharpe": observed_sharpe,
        "n_trials": n_trials,
        "sharpe_std": sharpe_std,
        "n_obs": n_obs,
    }

    # Layer 5a: Bootstrap
    if bootstrap_result is not None:
        data["bootstrap"] = {
            "n_simulations": bootstrap_result.n_simulations,
            "sharpe_ci_lower": bootstrap_result.sharpe_ci_lower,
            "sharpe_ci_upper": bootstrap_result.sharpe_ci_upper,
            "sharpe_mean": bootstrap_result.sharpe_mean,
            "maxdd_median": bootstrap_result.maxdd_median,
            "maxdd_95th": bootstrap_result.maxdd_95th,
            "verdict": bootstrap_result.verdict,
        }
    else:
        data["bootstrap"] = None

    # Layer 5b: Permutation
    if perm_result is not None:
        data["permutation"] = {
            "n_permutations": perm_result.n_permutations,
            "real_sharpe": perm_result.real_sharpe,
            "p_value": perm_result.p_value,
            "verdict": perm_result.verdict,
        }
    else:
        data["permutation"] = None

    # Layer 6a: Industrial
    if industrial_result is not None:
        data["industrial"] = {
            "basic_sharpe": industrial_result.basic_sharpe,
            "industrial_sharpe": industrial_result.industrial_sharpe,
            "decay_pct": industrial_result.decay_pct,
            "verdict": industrial_result.verdict,
        }
    else:
        data["industrial"] = None

    # Layer 6b: Stress
    if stress_result is not None:
        data["stress"] = {
            "slippage_sensitivity": stress_result.slippage_sensitivity,
            "slippage_decay_pct": stress_result.slippage_decay_pct,
            "cost_doubled_sharpe": stress_result.cost_doubled_sharpe,
            "adjacent_freq_sharpe": stress_result.adjacent_freq_sharpe,
            "similar_instrument_sharpe": stress_result.similar_instrument_sharpe,
        }
    else:
        data["stress"] = None

    # Final verdict
    if val_pipeline_result is not None:
        data["verdict"] = {
            "hard_reject": val_pipeline_result.hard_reject,
            "reject_reasons": list(val_pipeline_result.reject_reasons),
            "soft_flags": list(val_pipeline_result.soft_flags),
        }
    else:
        data["verdict"] = {
            "hard_reject": False,
            "reject_reasons": [],
            "soft_flags": [],
        }

    return data


# ── 3. attribution.md ────────────────────────────────────────────────────────

def generate_attribution_md(
    *,
    signal_result: SignalAttributionResult | None = None,
    horizon_result: HorizonAttributionResult | None = None,
    regime_result: RegimeAttributionResult | None = None,
    baseline_result: BaselineDecomposition | None = None,
    operational_result: OperationalAttribution | None = None,
    strategy_name: str = "",
    symbol: str = "",
) -> str:
    """Generate complete attribution.md with all 5 layers.

    Delegates to the existing generate_attribution_report() which already
    handles all 5 layers when results are provided.
    """
    return generate_attribution_report(
        signal_result=signal_result,
        horizon_result=horizon_result,
        regime_result=regime_result,
        baseline_result=baseline_result,
        operational_result=operational_result,
        strategy_name=strategy_name,
        symbol=symbol,
    )


# ── Unified orchestrator ─────────────────────────────────────────────────────

def generate_companion_files(
    output_dir: Path,
    *,
    # params.yaml inputs
    strategy_class: type,
    best_params: dict[str, Any],
    opt_score: float = 0.0,
    is_robust: bool = False,
    symbol: str = "",
    freq: str = "daily",
    n_trials: int = 0,
    # validation.yaml inputs
    cv_result: RegimeCVResult | None = None,
    oos_result: OOSResult | None = None,
    oos_full_span: dict | None = None,
    oos_period_breakdown: list[dict] | None = None,
    period_concentration_warning: bool = False,
    wf_result: WalkForwardResult | None = None,
    dsr_value: float | None = None,
    observed_sharpe: float | None = None,
    dsr_n_trials: int | None = None,
    sharpe_std: float | None = None,
    n_obs: int | None = None,
    bootstrap_result: BootstrapResult | None = None,
    perm_result: PermutationResult | None = None,
    industrial_result: IndustrialResult | None = None,
    stress_result: StressTestResult | None = None,
    val_pipeline_result: ValidationPipelineResult | None = None,
    # attribution.md inputs
    signal_result: SignalAttributionResult | None = None,
    horizon_result: HorizonAttributionResult | None = None,
    regime_result: RegimeAttributionResult | None = None,
    baseline_result: BaselineDecomposition | None = None,
    operational_result: OperationalAttribution | None = None,
) -> None:
    """Generate all 3 companion files in one call."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_name = getattr(strategy_class, "name", strategy_class.__name__)

    # 1. params.yaml
    params_data = generate_params_yaml(
        strategy_class, best_params, opt_score, is_robust,
        symbol, freq, n_trials,
    )
    _save_yaml(params_data, output_dir / "params.yaml")

    # 2. validation.yaml
    val_data = generate_validation_yaml(
        cv_result=cv_result,
        oos_result=oos_result,
        oos_full_span=oos_full_span,
        oos_period_breakdown=oos_period_breakdown,
        period_concentration_warning=period_concentration_warning,
        wf_result=wf_result,
        dsr_value=dsr_value,
        observed_sharpe=observed_sharpe,
        n_trials=dsr_n_trials,
        sharpe_std=sharpe_std,
        n_obs=n_obs,
        bootstrap_result=bootstrap_result,
        perm_result=perm_result,
        industrial_result=industrial_result,
        stress_result=stress_result,
        val_pipeline_result=val_pipeline_result,
    )
    _save_yaml(val_data, output_dir / "validation.yaml")

    # 3. attribution.md
    attr_md = generate_attribution_md(
        signal_result=signal_result,
        horizon_result=horizon_result,
        regime_result=regime_result,
        baseline_result=baseline_result,
        operational_result=operational_result,
        strategy_name=strategy_name,
        symbol=symbol,
    )
    (output_dir / "attribution.md").write_text(attr_md, encoding="utf-8")
