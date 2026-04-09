"""Batch regenerate all long/AG/1h strategies with full pipeline.

Uses ALL OOS periods (direction=up, split=oos, any regime).
Renames output folders using oos.html Total Return.
"""
from __future__ import annotations

import sys
import re
import shutil
from pathlib import Path

import numpy as np
import yaml

from pipeline.qbase_config import ALPHAFORGE_PATH as _AF_PATH, PROJECT_ROOT
_QB_PATH = "/Users/simon/Desktop/QBase_v2"
for p in (_AF_PATH, _QB_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util

from pipeline.backtest_runner import run_qbase_backtest
from alphaforge.report import HTMLReportGenerator
from alphaforge.data.market import MarketDataLoader
from regime.schema import load_labels
from validation.pipeline import run_validation_pipeline
from validation.deflated_sharpe import deflated_sharpe_ratio
from validation.regime_cv import run_regime_cv
from validation.oos_validator import validate_oos
from validation.walk_forward import walk_forward_verdict
from validation.permutation_test import permutation_test
from validation.industrial_check import check_industrial_decay
from validation.stress_test import run_stress_test
from attribution.horizon import horizon_attribution
from attribution.baseline import decompose_baseline
from attribution.operational import operational_attribution
from attribution.report import generate_attribution_report
from strategies.baselines.tsmom_fast import TSMOMFast
from strategies.baselines.tsmom_medium import TSMOMMedium
from strategies.baselines.tsmom_slow import TSMOMSlow

PROJECT_ROOT = Path(_QB_PATH)
SYMBOL = "AG"
FREQ = "1h"
DIRECTION = "long"
REGIME = "long"
STRATEGY_DIR = PROJECT_ROOT / "strategies" / "long" / "AG" / "1h"
RESEARCH_DIR = PROJECT_ROOT / "research" / "long" / "long" / "AG" / "1h"

loader = MarketDataLoader(f"{_AF_PATH}/data/")
reporter = HTMLReportGenerator()

# ── Load labels ──
regime_config = load_labels(PROJECT_ROOT / "data" / "regime_labels" / "AG_long.yaml")
train_labels = [l for l in regime_config.labels if l.split == "train" and l.direction == "up"]
oos_labels = [l for l in regime_config.labels if l.split == "oos" and l.direction == "up"]

train_active = [{"start": str(l.start), "end": str(l.end)} for l in train_labels]
oos_active = [{"start": str(l.start), "end": str(l.end)} for l in oos_labels]
train_start = str(min(l.buffer_start or l.start for l in train_labels))
train_end = str(max(l.buffer_end or l.end for l in train_labels))
oos_start = str(min(l.buffer_start or l.start for l in oos_labels))
oos_end = str(max(l.buffer_end or l.end for l in oos_labels))

# Pre-load bars (shared across all strategies)
train_bars = loader.load(SYMBOL, freq=FREQ, start=train_start, end=train_end)
oos_bars = loader.load(SYMBOL, freq=FREQ, start=oos_start, end=oos_end)


def _run_full(cls, params, labels, industrial=False, config_overrides=None):
    active = [{"start": str(l.start), "end": str(l.end)} for l in labels]
    s = str(min(l.buffer_start or l.start for l in labels))
    e = str(max(l.buffer_end or l.end for l in labels))
    return run_qbase_backtest(cls, params, SYMBOL, FREQ, start=s, end=e,
                              direction=DIRECTION, active_periods=active,
                              industrial=industrial, config_overrides=config_overrides)


def _dr(r):
    d = r.daily_returns
    return np.asarray(d.values if hasattr(d, "values") else d, dtype=np.float64)


def _avg_hold(r):
    try:
        t = r.trades
        if t is not None and len(t) > 0 and "hold_bars" in t.columns:
            return float(t["hold_bars"].mean())
    except Exception:
        pass
    return 0.0


def run_strategy_pipeline(version: str, strategy_file: Path) -> dict:
    """Run full pipeline for one strategy version."""
    print(f"\n{'═' * 60}")
    print(f"  {SYMBOL} {DIRECTION.upper()} │ {REGIME} │ {FREQ} │ {version}")
    print(f"{'═' * 60}")

    # Import strategy
    spec = importlib.util.spec_from_file_location(version, str(strategy_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls_name = [n for n in dir(mod) if n.startswith("StrongTrend")][0]
    StrategyCls = getattr(mod, cls_name)
    strategy_name = getattr(StrategyCls, "name", cls_name)

    best_params = {}
    temp_dir = RESEARCH_DIR / f"_{version}_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # B3: Bare Logic Gate
        bare = _run_full(StrategyCls, {}, train_labels)
        print(f"  [B3] n_trades={bare.n_trades}  pf={getattr(bare,'profit_factor',0):.2f}  sharpe={bare.sharpe:.2f}")

        # D1: Regime CV (LOO)
        fold_sharpes = []
        for i in range(len(train_labels)):
            loo = [{"start": str(l.start), "end": str(l.end)} for j, l in enumerate(train_labels) if j != i]
            try:
                r = run_qbase_backtest(StrategyCls, {}, SYMBOL, FREQ, start=train_start, end=train_end,
                                       direction=DIRECTION, active_periods=loo)
                fold_sharpes.append(r.sharpe)
            except Exception:
                pass
        if fold_sharpes:
            cv = run_regime_cv(fold_sharpes, strategy_name, REGIME)
            print(f"  [D1] cv={cv.verdict}  mean={cv.mean_sharpe:.2f}")

        # D2: OOS + IS
        oos_result = _run_full(StrategyCls, {}, oos_labels)
        is_result = _run_full(StrategyCls, {}, train_labels)
        oos_sharpe = oos_result.sharpe
        is_sharpe = is_result.sharpe
        oos_daily = _dr(oos_result)
        print(f"  [D2] is={is_sharpe:.2f}  oos={oos_sharpe:.2f}  n_trades={oos_result.n_trades}")

        # D3-D4
        wf = walk_forward_verdict(fold_sharpes, "regime_aware") if fold_sharpes else None
        sharpe_std = float(np.std(fold_sharpes)) if len(fold_sharpes) > 1 else 0.0
        dsr = deflated_sharpe_ratio(oos_sharpe, 1, sharpe_std, max(len(oos_daily), 1))

        # D5b
        perm = permutation_test(oos_daily, oos_sharpe) if len(oos_daily) >= 30 and oos_sharpe != 0 else None

        # D6a: Industrial
        ind = _run_full(StrategyCls, {}, oos_labels, industrial=True)
        industrial_sharpe = ind.sharpe
        ind_check = check_industrial_decay(oos_sharpe, industrial_sharpe)

        # D6b: Stress
        stress = _run_full(StrategyCls, {}, oos_labels, config_overrides={"slippage_ticks": 2.0})
        stress_sharpe = stress.sharpe
        stress_obj = run_stress_test(oos_sharpe, stress_sharpe)
        print(f"  [D6] industrial={industrial_sharpe:.2f}  stress={stress_sharpe:.2f}  sens={stress_obj.slippage_sensitivity}")

        # Validation pipeline
        skew, kurt = 0.0, 3.0
        if len(oos_daily) > 3:
            m, s = np.mean(oos_daily), np.std(oos_daily)
            if s > 0:
                skew = float(np.mean(((oos_daily - m) / s) ** 3))
                kurt = float(np.mean(((oos_daily - m) / s) ** 4))

        val = run_validation_pipeline(
            fold_sharpes=fold_sharpes, strategy=strategy_name, regime=REGIME,
            is_sharpe=is_sharpe, oos_sharpe=oos_sharpe, is_trades=is_result.n_trades,
            oos_trades=oos_result.n_trades, is_avg_hold=_avg_hold(is_result),
            oos_avg_hold=_avg_hold(oos_result), window_sharpes=fold_sharpes,
            wf_mode="regime_aware", observed_sharpe=oos_sharpe, n_trials=1,
            sharpe_std=sharpe_std, n_obs=max(len(oos_daily), 1), skew=skew, kurt=kurt,
            daily_returns=oos_daily if len(oos_daily) >= 30 else None,
            basic_sharpe=oos_sharpe, industrial_sharpe=industrial_sharpe,
            doubled_slippage_sharpe=stress_sharpe,
        )
        print(f"  [val] hard_reject={val.hard_reject}  soft_flags={list(val.soft_flags)}")

        # Per-period breakdown
        period_returns = []
        for lbl in oos_labels:
            pr = run_qbase_backtest(StrategyCls, {}, SYMBOL, FREQ,
                                     start=str(lbl.start), end=str(lbl.end), direction=DIRECTION)
            period_returns.append({
                "period": f"{lbl.start} → {lbl.end}", "regime": lbl.regime,
                "total_return": float(pr.total_return), "sharpe": float(pr.sharpe),
                "n_trades": int(pr.n_trades),
            })

        # Generate reports
        reporter.generate(is_result, str(temp_dir / "train.html"), bar_data={SYMBOL: train_bars}, freq=FREQ)
        reporter.generate(oos_result, str(temp_dir / "oos.html"), bar_data={SYMBOL: oos_bars}, freq=FREQ)

        # Extract return from oos.html
        html = (temp_dir / "oos.html").read_text(encoding="utf-8")
        m = re.search(r'总收益.*?<div[^>]*class="value[^"]*"[^>]*>([-+]?\d+\.\d+)%', html, re.DOTALL)
        oos_return_html = float(m.group(1)) if m else oos_result.total_return * 100
        print(f"  [html] Total Return = {oos_return_html:.2f}%")

        # Attribution
        try:
            fast_r, med_r, slow_r = (_run_full(c, {}, oos_labels) for c in (TSMOMFast, TSMOMMedium, TSMOMSlow))
            t1, t3, t12 = _dr(fast_r), _dr(med_r), _dr(slow_r)
            ml = min(len(oos_daily), len(t1), len(t3), len(t12))
            h_res = horizon_attribution(oos_daily[:ml], t1[:ml], t3[:ml], t12[:ml]) if ml >= 10 else None
            bl_res = decompose_baseline(oos_daily[:min(len(oos_daily), len(t3))],
                                         t3[:min(len(oos_daily), len(t3))]) if min(len(oos_daily), len(t3)) >= 10 else None
            op_res = operational_attribution(basic_sharpe=float(oos_sharpe), industrial_sharpe=float(industrial_sharpe))
            attr_md = generate_attribution_report(horizon_result=h_res, regime_result=None,
                                                   baseline_result=bl_res, operational_result=op_res,
                                                   strategy_name=strategy_name, symbol=SYMBOL)
            (temp_dir / "attribution.md").write_text(attr_md, encoding="utf-8")
        except Exception as e:
            print(f"  [attr] error: {e}")

        # Save yamls
        val_summary = {
            "hard_reject": val.hard_reject,
            "reject_reasons": list(val.reject_reasons),
            "soft_flags": list(val.soft_flags),
            "oos_full_span": {
                "start": oos_start, "end": oos_end,
                "total_return": float(oos_result.total_return),
                "annualized_return": float(oos_result.annualized_return),
                "sharpe": float(oos_sharpe), "n_trades": int(oos_result.n_trades),
                "max_drawdown": float(oos_result.max_drawdown),
                "profit_factor": float(getattr(oos_result, "profit_factor", 0.0)),
            },
            "oos_period_breakdown": period_returns,
            "oos_sharpe": float(oos_sharpe),
            "industrial_oos_sharpe": float(industrial_sharpe),
            "is_sharpe": float(is_sharpe),
            "n_trades_oos": int(oos_result.n_trades),
            "dsr": float(dsr),
        }
        with open(temp_dir / "validation.yaml", "w") as f:
            yaml.dump(val_summary, f, default_flow_style=False)
        with open(temp_dir / "params.yaml", "w") as f:
            yaml.dump({"best_params": {}, "opt_score": float(is_sharpe), "is_robust": True}, f, default_flow_style=False)

        # Rename folder
        sign = "+" if oos_return_html >= 0 else ""
        final_name = f"{version}_{sign}{oos_return_html:.2f}%"
        final_dir = RESEARCH_DIR / final_name
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.rename(final_dir)
        print(f"  ✓ {final_name}")

        return {"version": version, "folder": final_name, "oos_return": oos_return_html,
                "oos_sharpe": oos_sharpe, "hard_reject": val.hard_reject}

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return {"version": version, "error": str(e)}


# ── Main ──
if __name__ == "__main__":
    strategy_files = sorted(STRATEGY_DIR.glob("v*.py"))
    results = []

    for sf in strategy_files:
        # Extract version: v1, v2, ..., v10
        ver = sf.stem.split("_")[-1]  # e.g. "v10"
        results.append(run_strategy_pipeline(ver, sf))

    # Clean up old folders
    print(f"\n{'═' * 60}")
    print("  Cleaning old folders...")
    new_folders = {r["folder"] for r in results if "folder" in r}
    for old in RESEARCH_DIR.iterdir():
        if old.is_dir() and old.name not in new_folders and not old.name.startswith("_"):
            print(f"  removing: {old.name}")
            shutil.rmtree(old)

    # Summary
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}")
    for r in sorted(results, key=lambda x: x.get("oos_return", -999), reverse=True):
        if "error" in r:
            print(f"  {r['version']:>4}  ERROR: {r['error']}")
        else:
            reject = " REJECTED" if r["hard_reject"] else ""
            print(f"  {r['version']:>4}  {r['folder']:<25}  sharpe={r['oos_sharpe']:.2f}{reject}")
