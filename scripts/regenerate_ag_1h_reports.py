"""Regenerate train.html + oos.html for ALL long/AG/1h strategies.

Uses AlphaForge V7.6.1 report system. Updates folder names with OOS total return.
"""
from __future__ import annotations

import importlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from pipeline.qbase_config import ALPHAFORGE_PATH, PROJECT_ROOT, DATA_DIR

from alphaforge.data.market import MarketDataLoader
from alphaforge.data.contract_specs import ContractSpecManager
from alphaforge.report import HTMLReportGenerator
from pipeline.backtest_runner import run_qbase_backtest
from regime.schema import load_labels

# ── Constants ─────────────────────────────────────────────────────────────────
SYMBOL = "AG"
DIRECTION = "long"
REGIME = "long"
FREQ = "1h"
SIGNAL_DIRECTION = "long"

STRATEGY_DIR = PROJECT_ROOT / "strategies" / REGIME / DIRECTION / SYMBOL / FREQ
RESEARCH_DIR = PROJECT_ROOT / "research" / REGIME / DIRECTION / SYMBOL / FREQ
LABEL_PATH = PROJECT_ROOT / "data" / "regime_labels" / f"{SYMBOL}.yaml"

# Direction mapping for regime labels
DIR_MAP = {"long": "up", "short": "down"}


def load_periods():
    """Load train and OOS periods from regime labels.

    Key rules (matching original batch_regenerate_ag_1h.py):
    - active_periods use CORE dates (lbl.start/lbl.end) — strategy trades only here
    - bar loading uses BUFFER dates (lbl.buffer_start/lbl.buffer_end) — wider for warmup
    - Train: filter by regime type (long only)
    - OOS: ALL oos periods for this direction (not filtered by regime)
    """
    config = load_labels(LABEL_PATH)
    af_dir = DIR_MAP[DIRECTION]

    train_labels = [l for l in config.labels
                    if l.split == "train" and l.direction == af_dir and l.regime == REGIME]
    oos_labels = [l for l in config.labels
                  if l.split == "oos" and l.direction == af_dir]

    # Active periods = core dates (where strategy actually trades)
    train_active = [{"start": str(l.start), "end": str(l.end)} for l in train_labels]
    oos_active = [{"start": str(l.start), "end": str(l.end)} for l in oos_labels]

    # Bar loading range = buffer dates (wider, for indicator warmup)
    train_range = {
        "start": str(min(l.buffer_start or l.start for l in train_labels)),
        "end": str(max(l.buffer_end or l.end for l in train_labels)),
    } if train_labels else None
    oos_range = {
        "start": str(min(l.buffer_start or l.start for l in oos_labels)),
        "end": str(max(l.buffer_end or l.end for l in oos_labels)),
    } if oos_labels else None

    return train_active, oos_active, train_range, oos_range


def load_bars_for_range(date_range):
    """Load BarArray for a date range dict with start/end."""
    if not date_range:
        return None
    loader = MarketDataLoader(DATA_DIR)
    bars = loader.load(SYMBOL, freq=FREQ, start=date_range["start"], end=date_range["end"])
    return {SYMBOL: bars}


def discover_strategies():
    """Find all v*.py strategy files and import their classes."""
    strategies = []
    for f in sorted(STRATEGY_DIR.glob("*.py")):
        if f.name == "__init__.py":
            continue

        # Extract version number
        name = f.stem  # e.g., "long_AG_1h_v1"
        version = name.split("_v")[-1] if "_v" in name else name.replace("v", "")

        # Dynamic import
        module_path = f"strategies.{REGIME}.{DIRECTION}.{SYMBOL}.{FREQ}.{f.stem}"
        try:
            if module_path in sys.modules:
                del sys.modules[module_path]
            mod = importlib.import_module(module_path)

            # Find the strategy class
            strategy_class = None
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (isinstance(obj, type)
                    and hasattr(obj, "name")
                    and attr_name != "TrendingStrategy"
                    and not attr_name.startswith("_")):
                    strategy_class = obj
                    break

            if strategy_class is None:
                print(f"  [skip] {f.name}: no strategy class found")
                continue

            strategies.append({
                "version": f"v{version}",
                "filename": f.name,
                "class": strategy_class,
            })
        except Exception as e:
            print(f"  [skip] {f.name}: import error: {e}")

    return strategies


def find_research_dir(version: str) -> Path | None:
    """Find existing research directory for a version (v{N}_... pattern)."""
    import re
    pattern = re.compile(rf"^{version}_")
    for entry in RESEARCH_DIR.iterdir():
        if entry.is_dir() and pattern.match(entry.name):
            return entry
    return None


def load_params(research_dir: Path) -> dict:
    """Load optimized parameters from params.yaml."""
    params_path = research_dir / "params.yaml"
    if not params_path.exists():
        return {}
    with open(params_path) as f:
        data = yaml.safe_load(f)
    return data.get("best_params", {}) if data else {}


def run_strategy(strategy_info, train_active, oos_active, train_range, oos_range, train_bars, oos_bars):
    """Run backtest and generate reports for a single strategy."""
    version = strategy_info["version"]
    cls = strategy_info["class"]
    t0 = time.time()

    print(f"\n{'─'*50}")
    print(f"  {version}: {cls.__name__}")

    # Find existing research dir
    research_dir = find_research_dir(version)
    if research_dir is None:
        # Create new directory (temporary name)
        research_dir = RESEARCH_DIR / f"{version}_pending"
        research_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [new] {research_dir.name}")

    # Load params
    params = load_params(research_dir)
    print(f"  params: {params or '(defaults)'}")

    # ── Train backtest + report ───────────────────────────────────────────
    try:
        train_result = run_qbase_backtest(
            cls, params, symbol=SYMBOL, freq=FREQ,
            start=train_range["start"],
            end=train_range["end"],
            direction=SIGNAL_DIRECTION,
            active_periods=train_active,
            industrial=True,
        )
        print(f"  [train] Sharpe={train_result.sharpe:.2f}  Return={train_result.total_return*100:.2f}%")

        reporter = HTMLReportGenerator()
        train_path = research_dir / "train.html"
        reporter.generate(train_result, str(train_path), bar_data=train_bars, freq=FREQ)
        print(f"  [train] Report saved: {train_path.name}")
    except Exception as e:
        print(f"  [train] FAILED: {e}")
        train_result = None

    # ── OOS backtest + report ─────────────────────────────────────────────
    oos_result = None
    try:
        oos_result = run_qbase_backtest(
            cls, params, symbol=SYMBOL, freq=FREQ,
            start=oos_range["start"],
            end=oos_range["end"],
            direction=SIGNAL_DIRECTION,
            active_periods=oos_active,
            industrial=True,
        )
        print(f"  [oos]   Sharpe={oos_result.sharpe:.2f}  Return={oos_result.total_return*100:.2f}%")

        reporter = HTMLReportGenerator()
        oos_path = research_dir / "oos.html"
        reporter.generate(oos_result, str(oos_path), bar_data=oos_bars, freq=FREQ)
        print(f"  [oos]   Report saved: {oos_path.name}")
    except Exception as e:
        print(f"  [oos]   FAILED: {e}")

    # ── Rename folder with OOS return (extract from oos.html) ─────────────
    import re as _re
    oos_return_pct = None
    oos_html_path = research_dir / "oos.html"
    if oos_html_path.exists():
        try:
            html_content = oos_html_path.read_text(encoding="utf-8")
            m = _re.search(
                r'总收益.*?<div[^>]*class="value[^"]*"[^>]*>([-+]?\d+\.\d+)%',
                html_content, _re.DOTALL,
            )
            if m:
                oos_return_pct = float(m.group(1))
        except Exception:
            pass

    if oos_return_pct is not None:
        sign = "+" if oos_return_pct >= 0 else ""
        new_name = f"{version}_{sign}{oos_return_pct:.2f}%"
    else:
        new_name = version

    new_dir = RESEARCH_DIR / new_name
    if new_dir != research_dir:
        # Remove old directory with same version but different return
        if new_dir.exists():
            shutil.rmtree(new_dir)
        research_dir.rename(new_dir)
        print(f"  [rename] {research_dir.name} → {new_name}")

        # Clean up old folders for this version
        import re
        for old_dir in RESEARCH_DIR.iterdir():
            if (old_dir.is_dir()
                and old_dir.name.startswith(f"{version}_")
                and old_dir != new_dir):
                shutil.rmtree(old_dir)
                print(f"  [cleanup] removed {old_dir.name}")

    elapsed = time.time() - t0
    return {
        "version": version,
        "class": cls.__name__,
        "train_sharpe": train_result.sharpe if train_result else None,
        "oos_sharpe": oos_result.sharpe if oos_result else None,
        "oos_return": oos_return_pct,
        "folder": new_name,
        "elapsed": elapsed,
    }


def main():
    print(f"{'='*60}")
    print(f"  Regenerate Reports: {REGIME}/{DIRECTION}/{SYMBOL}/{FREQ}")
    print(f"  AlphaForge V7.6.1 | Industrial Mode")
    print(f"{'='*60}")

    # Load periods
    train_active, oos_active, train_range, oos_range = load_periods()
    print(f"\nTrain active periods: {len(train_active)}")
    for p in train_active:
        print(f"  {p['start']} → {p['end']}")
    print(f"OOS active periods: {len(oos_active)}")
    for p in oos_active:
        print(f"  {p['start']} → {p['end']}")
    print(f"Train bar range: {train_range}")
    print(f"OOS bar range:   {oos_range}")

    if not train_active or not oos_active:
        print("ERROR: No train or OOS periods found!")
        return

    # Pre-load bars (one-time, shared across all strategies)
    print("\nLoading bar data...")
    train_bars = load_bars_for_range(train_range)
    oos_bars = load_bars_for_range(oos_range)
    print(f"  Train bars loaded: {train_bars is not None}")
    print(f"  OOS bars loaded:   {oos_bars is not None}")

    # Discover strategies
    strategies = discover_strategies()
    print(f"\nFound {len(strategies)} strategies")

    # Run all
    results = []
    for s in strategies:
        r = run_strategy(s, train_active, oos_active, train_range, oos_range, train_bars, oos_bars)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Version':<10} {'OOS Sharpe':>12} {'OOS Return':>12} {'Folder':<30} {'Time':>8}")
    print(f"{'─'*10} {'─'*12} {'─'*12} {'─'*30} {'─'*8}")
    for r in results:
        sharpe = f"{r['oos_sharpe']:.2f}" if r["oos_sharpe"] is not None else "FAIL"
        ret = f"{r['oos_return']:.2f}%" if r["oos_return"] is not None else "FAIL"
        print(f"{r['version']:<10} {sharpe:>12} {ret:>12} {r['folder']:<30} {r['elapsed']:>7.1f}s")

    passed = sum(1 for r in results if r["oos_sharpe"] is not None)
    print(f"\nTotal: {len(results)} | Completed: {passed} | Failed: {len(results) - passed}")


if __name__ == "__main__":
    main()
