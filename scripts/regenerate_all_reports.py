"""Regenerate train.html + oos.html for ALL existing strategies with AlphaForge V7.2.

Scans research/ for all strategy versions that have params.yaml,
re-runs backtest with Industrial mode, generates fresh V7.2 reports,
and updates folder names with OOS return.

Usage:
    PYTHONPATH=.:../AlphaForge python scripts/regenerate_all_reports.py
    PYTHONPATH=.:../AlphaForge python scripts/regenerate_all_reports.py --group mild_trend/long/I
    PYTHONPATH=.:../AlphaForge python scripts/regenerate_all_reports.py --freq 1h
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
import time
from pathlib import Path

import yaml

from pipeline.qbase_config import PROJECT_ROOT, DATA_DIR
from pipeline.backtest_runner import run_qbase_backtest
from alphaforge.data.market import MarketDataLoader
from alphaforge.report import HTMLReportGenerator
from regime.schema import load_labels

DIR_MAP = {"long": "up", "short": "down"}
TIMEFRAMES = ["daily", "1h", "2h", "4h"]

# Label file mapping
LABEL_FILES = {
    ("mild_trend", "long", "I"): "I.yaml",
    ("mild_trend", "short", "I"): "I_short.yaml",
    ("strong_trend", "long", "AG"): "AG_long.yaml",
    ("strong_trend", "short", "AG"): "AG_short.yaml",
}


def discover_groups():
    """Scan strategies/ to find all regime/direction/instrument groups."""
    groups = []
    strategies_root = PROJECT_ROOT / "strategies"
    for regime_dir in sorted(strategies_root.iterdir()):
        if not regime_dir.is_dir() or regime_dir.name.startswith((".", "_", "template", "baseline")):
            continue
        regime = regime_dir.name
        if regime == "mean_reversion":
            for inst_dir in sorted(regime_dir.iterdir()):
                if inst_dir.is_dir() and not inst_dir.name.startswith("."):
                    groups.append((regime, "both", inst_dir.name))
        else:
            for dir_dir in sorted(regime_dir.iterdir()):
                if not dir_dir.is_dir() or dir_dir.name.startswith("."):
                    continue
                direction = dir_dir.name
                for inst_dir in sorted(dir_dir.iterdir()):
                    if inst_dir.is_dir() and not inst_dir.name.startswith("."):
                        groups.append((regime, direction, inst_dir.name))
    return groups


def regenerate_group(regime, direction, instrument, freq_filter=None):
    """Regenerate all reports for one group."""
    label_key = (regime, direction, instrument)
    label_file = LABEL_FILES.get(label_key)
    if not label_file:
        print(f"  [skip] No label file for {label_key}")
        return []

    label_path = PROJECT_ROOT / "data" / "regime_labels" / label_file
    if not label_path.exists():
        print(f"  [skip] Label file not found: {label_path}")
        return []

    config = load_labels(label_path)
    af_dir = DIR_MAP.get(direction, direction)

    train_labels = [l for l in config.labels if l.split == "train" and l.direction == af_dir]
    oos_labels = [l for l in config.labels if l.split == "oos" and l.direction == af_dir]

    if not train_labels or not oos_labels:
        print(f"  [skip] No train or OOS labels")
        return []

    train_active = [{"start": str(l.start), "end": str(l.end)} for l in train_labels]
    oos_active = [{"start": str(l.start), "end": str(l.end)} for l in oos_labels]
    train_start = str(min(l.buffer_start or l.start for l in train_labels))
    train_end = str(max(l.buffer_end or l.end for l in train_labels))
    oos_start = str(min(l.buffer_start or l.start for l in oos_labels))
    oos_end = str(max(l.buffer_end or l.end for l in oos_labels))

    loader = MarketDataLoader(DATA_DIR)
    reporter = HTMLReportGenerator()
    signal_dir = direction if direction != "both" else None
    results = []

    for freq in TIMEFRAMES:
        if freq_filter and freq != freq_filter:
            continue

        strategy_dir = PROJECT_ROOT / "strategies" / regime
        if direction != "both":
            strategy_dir = strategy_dir / direction
        strategy_dir = strategy_dir / instrument / freq

        research_dir = PROJECT_ROOT / "research" / regime
        if direction != "both":
            research_dir = research_dir / direction
        research_dir = research_dir / instrument / freq

        if not strategy_dir.exists():
            continue

        # Load bars once per freq
        try:
            train_bars = {instrument: loader.load(instrument, freq=freq, start=train_start, end=train_end)}
            oos_bars = {instrument: loader.load(instrument, freq=freq, start=oos_start, end=oos_end)}
        except Exception as e:
            print(f"  [{freq}] bar load failed: {e}")
            continue

        # Find all strategy files
        for f in sorted(strategy_dir.glob("v*.py")):
            if f.name == "__init__.py":
                continue
            m = re.match(r"v(\d+)\.py$", f.name)
            if not m:
                continue
            version = f"v{m.group(1)}"

            # Import strategy class
            mod_name = f"regen_{f.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, f)
            mod = importlib.util.module_from_spec(spec)
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                print(f"  [{freq}/{version}] import error: {e}")
                results.append({"version": version, "freq": freq, "status": "IMPORT_FAIL"})
                continue

            cls = None
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if (isinstance(obj, type) and hasattr(obj, 'name')
                    and attr not in ('TrendingStrategy', 'QBaseStrategy', 'MeanReversionStrategy')):
                    cls = obj
                    break
            if cls is None:
                continue

            # Find existing research dir
            existing_dir = None
            for d in research_dir.iterdir() if research_dir.exists() else []:
                if d.is_dir() and d.name.startswith(f"{version}_"):
                    existing_dir = d
                    break

            if existing_dir is None:
                existing_dir = research_dir / f"{version}_pending"
                existing_dir.mkdir(parents=True, exist_ok=True)

            # Load params (support both old and new format)
            params = {}
            params_file = existing_dir / "params.yaml"
            if params_file.exists():
                with open(params_file) as pf:
                    pdata = yaml.safe_load(pf) or {}
                    # New format: parameters.{name}.value
                    if "parameters" in pdata and isinstance(pdata["parameters"], dict):
                        params = {k: v["value"] for k, v in pdata["parameters"].items()
                                  if isinstance(v, dict) and "value" in v}
                    # Old format: best_params
                    elif "best_params" in pdata:
                        params = pdata.get("best_params", {})

            t0 = time.time()

            try:
                # Train report
                tr = run_qbase_backtest(cls, params, symbol=instrument, freq=freq,
                    start=train_start, end=train_end, direction=signal_dir,
                    active_periods=train_active, industrial=True)
                reporter.generate(tr, str(existing_dir / "train.html"),
                    bar_data=train_bars, freq=freq)

                # OOS report
                oos_r = run_qbase_backtest(cls, params, symbol=instrument, freq=freq,
                    start=oos_start, end=oos_end, direction=signal_dir,
                    active_periods=oos_active, industrial=True)
                reporter.generate(oos_r, str(existing_dir / "oos.html"),
                    bar_data=oos_bars, freq=freq)

                # Generate companion files (params.yaml with complete metadata)
                try:
                    from pipeline.report_files import generate_params_yaml, _save_yaml as _save_clean
                    params_full = generate_params_yaml(cls, params, 0.0, True, instrument, freq)
                    _save_clean(params_full, existing_dir / "params.yaml")
                except Exception as e:
                    print(f"    [companion] params.yaml error: {e}")

                # Extract OOS return from HTML
                oos_return_pct = None
                html_path = existing_dir / "oos.html"
                try:
                    html = html_path.read_text(encoding="utf-8")
                    match = re.search(r'总收益.*?<div[^>]*class="value[^"]*"[^>]*>([-+]?\d+\.\d+)%', html, re.DOTALL)
                    if match:
                        oos_return_pct = float(match.group(1))
                except Exception:
                    pass

                if oos_return_pct is not None:
                    sign = "+" if oos_return_pct >= 0 else ""
                    new_name = f"{version}_{sign}{oos_return_pct:.2f}%"
                else:
                    ret = oos_r.total_return * 100
                    sign = "+" if ret >= 0 else ""
                    new_name = f"{version}_{sign}{ret:.2f}%"

                new_dir = research_dir / new_name
                if new_dir != existing_dir:
                    if new_dir.exists():
                        shutil.rmtree(new_dir)
                    existing_dir.rename(new_dir)

                elapsed = time.time() - t0
                print(f"  [{freq}/{version}] Sharpe={oos_r.sharpe:.2f} Ret={oos_return_pct or oos_r.total_return*100:.2f}% -> {new_name} ({elapsed:.1f}s)")
                results.append({"version": version, "freq": freq, "status": "OK",
                    "sharpe": oos_r.sharpe, "return": oos_return_pct, "folder": new_name})

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [{freq}/{version}] FAIL: {e} ({elapsed:.1f}s)")
                results.append({"version": version, "freq": freq, "status": f"FAIL: {e}"})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", help="e.g. mild_trend/long/I")
    parser.add_argument("--freq", choices=TIMEFRAMES)
    args = parser.parse_args()

    groups = discover_groups()
    if args.group:
        parts = args.group.split("/")
        groups = [g for g in groups if "/".join(g) == args.group]

    total_ok = 0
    total_fail = 0
    for regime, direction, instrument in groups:
        print(f"\n{'='*70}")
        print(f"  {regime}/{direction}/{instrument}")
        print(f"{'='*70}")
        results = regenerate_group(regime, direction, instrument, freq_filter=args.freq)
        ok = sum(1 for r in results if r["status"] == "OK")
        fail = sum(1 for r in results if r["status"] != "OK")
        total_ok += ok
        total_fail += fail
        print(f"  → {ok} OK, {fail} failed")

    print(f"\n{'='*70}")
    print(f"  GRAND TOTAL: {total_ok} OK, {total_fail} failed")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
