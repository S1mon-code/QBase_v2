"""Strategy Quality Score (SQS) — unified scoring for strategy selection.

SQS = weighted combination of 6 dimensions:
  1. OOS Performance (30%) — return, Sharpe, profit_factor
  2. Stability (20%) — fold Sharpe consistency, walk-forward pass rate
  3. Regime Robustness (15%) — regime CV verdict, fold win rate
  4. Risk Control (15%) — max drawdown, tail risk
  5. Cost Robustness (10%) — industrial decay (ABSOLUTE, not %), stress sensitivity
  6. Statistical Significance (10%) — DSR, bootstrap, permutation

Range: 0-100. Higher = better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Dimension weights
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "oos_performance": 0.30,
    "stability": 0.20,
    "regime_robustness": 0.15,
    "risk_control": 0.15,
    "cost_robustness": 0.10,
    "statistical_significance": 0.10,
}

# ---------------------------------------------------------------------------
# Kill-switch thresholds
# ---------------------------------------------------------------------------

KILL_THRESHOLDS = {
    "min_sqs": 10,
    "min_oos_sharpe": 0.0,
    "min_trades": 5,
    "max_drawdown": 0.40,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert value to float, returning *default* for None / non-numeric."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if not np.isfinite(f) else f
    except (TypeError, ValueError):
        return default


def _clip(score: float) -> float:
    return float(np.clip(score, 0.0, 100.0))


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    """Nested dict access: _get(d, 'oos', 'full_span', 'sharpe')."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


# ---------------------------------------------------------------------------
# Dimension scorers (each returns 0–100)
# ---------------------------------------------------------------------------

def _score_oos_performance(v: dict) -> float:
    """OOS return, Sharpe, profit_factor → 0-100."""
    sharpe = _safe_float(_get(v, "oos", "full_span", "sharpe"))
    if sharpe < 0:
        return 0.0

    total_return = _safe_float(_get(v, "oos", "full_span", "total_return"))
    profit_factor = _safe_float(_get(v, "oos", "full_span", "profit_factor"))

    # Sharpe: 0 → 0, 0.5 → 40, 1.0 → 70, 1.5 → 90, 2.0+ → 100
    s_sharpe = _clip(min(sharpe / 2.0, 1.0) * 100)

    # Return: 0% → 0, 20% → 50, 50%+ → 100
    s_return = _clip(min(max(total_return, 0.0) / 0.5, 1.0) * 100)

    # Profit factor: 1.0 → 0, 1.5 → 50, 2.0+ → 100
    pf_score = max(profit_factor - 1.0, 0.0) / 1.0
    s_pf = _clip(min(pf_score, 1.0) * 100)

    return _clip(0.50 * s_sharpe + 0.30 * s_return + 0.20 * s_pf)


def _score_stability(v: dict) -> float:
    """Fold Sharpe consistency + walk-forward pass."""
    fold_sharpes = _get(v, "regime_cv", "fold_sharpes", default=[])
    if not fold_sharpes or not isinstance(fold_sharpes, list):
        return 0.0

    arr = np.array([_safe_float(x) for x in fold_sharpes])
    if len(arr) == 0:
        return 0.0

    mean_s = float(np.mean(arr))
    std_s = float(np.std(arr)) if len(arr) > 1 else 0.0

    # Consistency: low std relative to mean → good
    # CV (coefficient of variation): lower is better
    if mean_s > 0 and std_s >= 0:
        cv = std_s / mean_s if mean_s > 0.01 else 10.0
        s_consistency = _clip((1.0 - min(cv, 2.0) / 2.0) * 100)
    else:
        s_consistency = 0.0

    # Walk-forward pass
    wf_passed = _get(v, "walk_forward", "passed", default=False)
    wf_win_rate = _safe_float(_get(v, "walk_forward", "win_rate"))
    s_wf = _clip(wf_win_rate * 100) if wf_passed else _clip(wf_win_rate * 50)

    return _clip(0.60 * s_consistency + 0.40 * s_wf)


def _score_regime_robustness(v: dict) -> float:
    """Regime CV verdict + fold win rate."""
    verdict = _get(v, "regime_cv", "verdict", default="FAIL")
    win_rate = _safe_float(_get(v, "regime_cv", "win_rate"))

    verdict_map = {"PASS": 100.0, "MARGINAL": 50.0, "FAIL": 0.0}
    s_verdict = verdict_map.get(str(verdict).upper(), 0.0)

    s_win = _clip(win_rate * 100)

    return _clip(0.50 * s_verdict + 0.50 * s_win)


def _score_risk_control(v: dict) -> float:
    """Max drawdown + tail risk (bootstrap maxdd_95th)."""
    max_dd = abs(_safe_float(_get(v, "oos", "full_span", "max_drawdown")))
    maxdd_95 = abs(_safe_float(_get(v, "bootstrap", "maxdd_95th")))

    # Drawdown: 0% → 100, 20% → 50, 40%+ → 0
    s_dd = _clip((1.0 - min(max_dd / 0.40, 1.0)) * 100)

    # Bootstrap tail: 0% → 100, 15% → 50, 30%+ → 0
    s_tail = _clip((1.0 - min(maxdd_95 / 0.30, 1.0)) * 100)

    return _clip(0.60 * s_dd + 0.40 * s_tail)


def _score_cost_robustness(v: dict) -> float:
    """Industrial decay (ABSOLUTE) + stress sensitivity.

    Uses abs(basic_sharpe - industrial_sharpe) instead of decay_pct,
    which fixes the wildly negative percentages when basic Sharpe ≈ 0.
    """
    basic_sharpe = _safe_float(_get(v, "industrial", "basic_sharpe"))
    industrial_sharpe = _get(v, "industrial", "industrial_sharpe")

    if industrial_sharpe is None:
        # Null industrial → neutral score (not penalized)
        s_decay = 50.0
    else:
        ind_s = _safe_float(industrial_sharpe)
        abs_decay = abs(basic_sharpe - ind_s)
        # abs_decay: 0 → 100, 0.25 → 50, 0.5+ → 0
        s_decay = _clip((1.0 - min(abs_decay / 0.5, 1.0)) * 100)

    # Stress sensitivity: LOW → 100, MEDIUM → 50, HIGH → 0
    stress = _get(v, "stress", "slippage_sensitivity", default="MEDIUM")
    stress_map = {"LOW": 100.0, "MEDIUM": 50.0, "HIGH": 0.0}
    s_stress = stress_map.get(str(stress).upper(), 50.0)

    return _clip(0.70 * s_decay + 0.30 * s_stress)


def _score_statistical_significance(v: dict) -> float:
    """DSR, bootstrap, permutation."""
    dsr = _safe_float(_get(v, "deflated_sharpe", "dsr"))
    bootstrap_verdict = _get(v, "bootstrap", "verdict", default="FRAGILE")
    perm = _get(v, "permutation")

    # DSR: 0 → 0, 0.5 → 25, 0.95 → 90, 1.0 → 100
    s_dsr = _clip(dsr * 100)

    # Bootstrap verdict
    boot_map = {"ROBUST": 100.0, "ACCEPTABLE": 60.0, "FRAGILE": 10.0}
    s_boot = boot_map.get(str(bootstrap_verdict).upper(), 10.0)

    # Permutation
    if perm is None:
        s_perm = 30.0  # neutral when missing
    else:
        p_verdict = _get(perm, "verdict", default="NOT_SIGNIFICANT")
        perm_map = {
            "SIGNIFICANT": 100.0,
            "MARGINAL": 50.0,
            "NOT_SIGNIFICANT": 0.0,
        }
        s_perm = perm_map.get(str(p_verdict).upper(), 0.0)

    return _clip(0.40 * s_dsr + 0.30 * s_boot + 0.30 * s_perm)


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

_SCORERS = {
    "oos_performance": _score_oos_performance,
    "stability": _score_stability,
    "regime_robustness": _score_regime_robustness,
    "risk_control": _score_risk_control,
    "cost_robustness": _score_cost_robustness,
    "statistical_significance": _score_statistical_significance,
}


def compute_sqs(validation_data: dict) -> dict:
    """Compute SQS from validation.yaml data.

    Returns
    -------
    dict with keys:
        sqs: float (0-100)
        dimensions: dict[str, float]
        kill: bool
        kill_reasons: list[str]
    """
    v = validation_data
    kill_reasons: list[str] = []

    # Hard overrides -------------------------------------------------------
    n_trades = int(_safe_float(_get(v, "oos", "full_span", "n_trades")))
    hard_reject = bool(_get(v, "verdict", "hard_reject", default=False))
    oos_sharpe = _safe_float(_get(v, "oos", "full_span", "sharpe"))
    max_dd = abs(_safe_float(_get(v, "oos", "full_span", "max_drawdown")))

    if n_trades == 0:
        kill_reasons.append("zero_trades")
    if hard_reject:
        kill_reasons.append("hard_reject")
    if oos_sharpe < KILL_THRESHOLDS["min_oos_sharpe"]:
        kill_reasons.append("negative_oos_sharpe")
    if n_trades < KILL_THRESHOLDS["min_trades"] and n_trades > 0:
        kill_reasons.append(f"too_few_trades({n_trades})")
    if max_dd > KILL_THRESHOLDS["max_drawdown"]:
        kill_reasons.append(f"excessive_drawdown({max_dd:.1%})")

    # Dimension scores -----------------------------------------------------
    dimensions = {}
    for name, scorer in _SCORERS.items():
        try:
            dimensions[name] = _clip(scorer(v))
        except Exception:
            dimensions[name] = 0.0

    # Weighted SQS ---------------------------------------------------------
    sqs = sum(dimensions[name] * WEIGHTS[name] for name in WEIGHTS)
    sqs = _clip(sqs)

    # Kill overrides -------------------------------------------------------
    if n_trades == 0 or hard_reject:
        sqs = 0.0
    if sqs < KILL_THRESHOLDS["min_sqs"]:
        if "low_sqs" not in kill_reasons:
            kill_reasons.append("low_sqs")

    kill = len(kill_reasons) > 0

    return {
        "sqs": round(sqs, 2),
        "dimensions": {k: round(v, 2) for k, v in dimensions.items()},
        "kill": kill,
        "kill_reasons": kill_reasons,
    }


# ---------------------------------------------------------------------------
# Scanning — adapted for QBase_v2 research/ directory structure
# ---------------------------------------------------------------------------

def _parse_strategy_path(yaml_path: Path) -> dict[str, str]:
    """Extract direction/instrument/freq/version from path.

    Expected v2 structure (post regime-simplification):
        research/{direction}/{instrument}/{freq}/v{N}_{return}%/validation.yaml
    """
    parts = yaml_path.parts
    # Find 'research' anchor
    try:
        idx = parts.index("research")
    except ValueError:
        return {}
    if idx + 4 >= len(parts):
        return {}
    return {
        "direction": parts[idx + 1],
        "instrument": parts[idx + 2],
        "freq": parts[idx + 3],
        "version_dir": parts[idx + 4],
        "quadrant": f"{parts[idx + 1]}_{parts[idx + 2]}",
    }


def scan_all_strategies(research_dir: Path) -> list[dict]:
    """Scan research/ and compute SQS for every strategy.

    Returns sorted list (highest SQS first).
    """
    results: list[dict] = []
    research_dir = Path(research_dir)

    for yaml_path in sorted(research_dir.rglob("validation.yaml")):
        meta = _parse_strategy_path(yaml_path)
        if not meta:
            continue

        try:
            with open(yaml_path, "r") as f:
                vdata = yaml.safe_load(f) or {}
        except Exception:
            continue

        sqs_result = compute_sqs(vdata)

        entry = {
            "path": str(yaml_path),
            **meta,
            **sqs_result,
            # Carry forward key OOS metrics for portfolio use
            "oos_sharpe": _safe_float(_get(vdata, "oos", "full_span", "sharpe")),
            "oos_return": _safe_float(_get(vdata, "oos", "full_span", "total_return")),
            "oos_max_dd": abs(
                _safe_float(_get(vdata, "oos", "full_span", "max_drawdown"))
            ),
            "oos_n_trades": int(
                _safe_float(_get(vdata, "oos", "full_span", "n_trades"))
            ),
            "oos_profit_factor": _safe_float(
                _get(vdata, "oos", "full_span", "profit_factor")
            ),
            "fold_sharpes": _get(vdata, "regime_cv", "fold_sharpes", default=[]),
        }
        results.append(entry)

    results.sort(key=lambda x: x["sqs"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

def apply_kill_switch(
    strategies: list[dict],
    config: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Split into survivors and killed.

    Parameters
    ----------
    strategies : list of dicts from scan_all_strategies
    config : optional overrides for kill thresholds

    Returns
    -------
    (survivors, killed) — both sorted by SQS descending.
    """
    cfg = {**KILL_THRESHOLDS, **(config or {})}

    survivors: list[dict] = []
    killed: list[dict] = []

    for s in strategies:
        if s["kill"]:
            killed.append(s)
        else:
            survivors.append(s)

    survivors.sort(key=lambda x: x["sqs"], reverse=True)
    killed.sort(key=lambda x: x["sqs"], reverse=True)
    return survivors, killed
