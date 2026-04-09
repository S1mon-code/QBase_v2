"""Batch optimize + validate + generate reports for newly created strategies.

Processes all 3 groups:
1. long/long/I (v41-v50 daily, v51-v60 1h, v31-v40 2h, v31-v40 4h)
2. short/short/I (v11-v20 × 4 freqs)
3. short/short/AG (v11-v20 × 4 freqs)

Usage:
    PYTHONPATH=.:../AlphaForge python scripts/batch_optimize_new.py
    PYTHONPATH=.:../AlphaForge python scripts/batch_optimize_new.py --group I_short
    PYTHONPATH=.:../AlphaForge python scripts/batch_optimize_new.py --group I_long --freq daily
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
import time
from pathlib import Path

from pipeline.qbase_config import PROJECT_ROOT, ALPHAFORGE_PATH, DATA_DIR
from pipeline.dev_pipeline import run_single_strategy_pipeline
from alphaforge.data.market import MarketDataLoader
from alphaforge.report import HTMLReportGenerator
from pipeline.backtest_runner import run_qbase_backtest
from regime.schema import load_labels

TIMEFRAMES = ["daily", "1h", "2h", "4h"]

# Group definitions: (regime, direction, instrument, label_file, signal_direction, version_ranges)
GROUPS = {
    "I_long": {
        "regime": "long",
        "direction": "long",
        "instrument": "I",
        "label_file": "I.yaml",
        "signal_direction": "long",
        "versions": {"daily": (41, 50), "1h": (51, 60), "2h": (31, 40), "4h": (31, 40)},
    },
    "I_short": {
        "regime": "short",
        "direction": "short",
        "instrument": "I",
        "label_file": "I_short.yaml",
        "signal_direction": "short",
        "versions": {"daily": (11, 20), "1h": (11, 20), "2h": (11, 20), "4h": (11, 20)},
    },
    "AG_short": {
        "regime": "short",
        "direction": "short",
        "instrument": "AG",
        "label_file": "AG_short.yaml",
        "signal_direction": "short",
        "versions": {"daily": (11, 20), "1h": (11, 20), "2h": (11, 20), "4h": (11, 20)},
    },
}

DIR_MAP = {"long": "up", "short": "down"}


def discover_strategy_class(filepath: Path):
    """Dynamically import a strategy file and return the strategy class."""
    mod_name = f"strategy_{filepath.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    if spec is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  [import error] {filepath.name}: {e}")
        return None

    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if (isinstance(obj, type)
            and hasattr(obj, 'name')
            and hasattr(obj, 'direction')
            and not attr_name.startswith('_')
            and attr_name not in ('TrendingStrategy', 'QBaseStrategy', 'MeanReversionStrategy')):
            return obj
    return None


def run_group(group_name: str, freq_filter: str | None = None):
    """Run optimization pipeline for a strategy group."""
    cfg = GROUPS[group_name]
    regime = cfg["regime"]
    direction = cfg["direction"]
    instrument = cfg["instrument"]
    label_file = cfg["label_file"]
    signal_dir = cfg["signal_direction"]

    print(f"\n{'='*70}")
    print(f"  GROUP: {group_name} ({regime}/{direction}/{instrument})")
    print(f"{'='*70}")

    # Load regime labels
    label_path = PROJECT_ROOT / "data" / "regime_labels" / label_file
    config = load_labels(label_path)
    af_dir = DIR_MAP[direction]

    train_labels = [l for l in config.labels if l.split == "train" and l.direction == af_dir]
    oos_labels = [l for l in config.labels if l.split == "oos" and l.direction == af_dir]

    train_active = [{"start": str(l.start), "end": str(l.end)} for l in train_labels]
    oos_active = [{"start": str(l.start), "end": str(l.end)} for l in oos_labels]
    train_range = {
        "start": str(min(l.buffer_start or l.start for l in train_labels)),
        "end": str(max(l.buffer_end or l.end for l in train_labels)),
    }
    oos_range = {
        "start": str(min(l.buffer_start or l.start for l in oos_labels)),
        "end": str(max(l.buffer_end or l.end for l in oos_labels)),
    }

    print(f"  Train: {len(train_active)} periods, OOS: {len(oos_active)} periods")

    # Pre-load bars
    loader = MarketDataLoader(DATA_DIR)
    reporter = HTMLReportGenerator()

    results = []

    for freq in TIMEFRAMES:
        if freq_filter and freq != freq_filter:
            continue

        v_start, v_end = cfg["versions"][freq]
        strategy_dir = PROJECT_ROOT / "strategies" / regime / direction / instrument / freq
        research_dir = PROJECT_ROOT / "research" / regime / direction / instrument / freq

        print(f"\n  --- {freq} (v{v_start}-v{v_end}) ---")

        # Load bars for this freq
        try:
            train_bars = loader.load(instrument, freq=freq, start=train_range["start"], end=train_range["end"])
            oos_bars = loader.load(instrument, freq=freq, start=oos_range["start"], end=oos_range["end"])
        except Exception as e:
            print(f"  [skip freq] Failed to load {freq} bars: {e}")
            continue

        for v in range(v_start, v_end + 1):
            filepath = strategy_dir / f"v{v}.py"
            if not filepath.exists():
                print(f"  [skip] v{v}.py not found")
                continue

            cls = discover_strategy_class(filepath)
            if cls is None:
                print(f"  [skip] v{v}.py: no strategy class")
                results.append({"version": f"v{v}", "freq": freq, "status": "IMPORT_FAIL"})
                continue

            t0 = time.time()
            print(f"\n  v{v}: {cls.__name__}")

            # Create research directory
            res_dir = research_dir / f"v{v}_pending"
            res_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Train backtest
                train_result = run_qbase_backtest(
                    cls, {}, symbol=instrument, freq=freq,
                    start=train_range["start"], end=train_range["end"],
                    direction=signal_dir, active_periods=train_active,
                    industrial=True,
                )
                print(f"    [train] Sharpe={train_result.sharpe:.2f} Return={train_result.total_return*100:.2f}%")

                # Train report
                reporter.generate(train_result, str(res_dir / "train.html"),
                    bar_data={instrument: train_bars}, freq=freq)

                # OOS backtest
                oos_result = run_qbase_backtest(
                    cls, {}, symbol=instrument, freq=freq,
                    start=oos_range["start"], end=oos_range["end"],
                    direction=signal_dir, active_periods=oos_active,
                    industrial=True,
                )
                print(f"    [oos]   Sharpe={oos_result.sharpe:.2f} Return={oos_result.total_return*100:.2f}%")

                # OOS report
                reporter.generate(oos_result, str(res_dir / "oos.html"),
                    bar_data={instrument: oos_bars}, freq=freq)

                # Rename folder with OOS return from HTML
                oos_return_pct = None
                oos_html = res_dir / "oos.html"
                if oos_html.exists():
                    try:
                        html = oos_html.read_text(encoding="utf-8")
                        m = re.search(
                            r'总收益.*?<div[^>]*class="value[^"]*"[^>]*>([-+]?\d+\.\d+)%',
                            html, re.DOTALL)
                        if m:
                            oos_return_pct = float(m.group(1))
                    except Exception:
                        pass

                if oos_return_pct is not None:
                    sign = "+" if oos_return_pct >= 0 else ""
                    new_name = f"v{v}_{sign}{oos_return_pct:.2f}%"
                else:
                    ret_pct = oos_result.total_return * 100
                    sign = "+" if ret_pct >= 0 else ""
                    new_name = f"v{v}_{sign}{ret_pct:.2f}%"

                new_dir = research_dir / new_name
                if new_dir.exists() and new_dir != res_dir:
                    shutil.rmtree(new_dir)
                if res_dir != new_dir:
                    res_dir.rename(new_dir)

                elapsed = time.time() - t0
                print(f"    [done] {new_name} ({elapsed:.1f}s)")

                results.append({
                    "version": f"v{v}", "freq": freq,
                    "status": "OK",
                    "oos_sharpe": oos_result.sharpe,
                    "oos_return": oos_return_pct or oos_result.total_return * 100,
                    "folder": new_name,
                    "elapsed": elapsed,
                })

            except Exception as e:
                elapsed = time.time() - t0
                print(f"    [FAIL] {e} ({elapsed:.1f}s)")
                results.append({"version": f"v{v}", "freq": freq, "status": f"FAIL: {e}"})

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {group_name}")
    print(f"{'='*70}")
    ok = [r for r in results if r["status"] == "OK"]
    fail = [r for r in results if r["status"] != "OK"]
    print(f"  Total: {len(results)} | OK: {len(ok)} | Failed: {len(fail)}")
    if ok:
        print(f"\n  {'Version':<8} {'Freq':<6} {'OOS Sharpe':>12} {'OOS Return':>12} {'Folder':<30}")
        for r in ok:
            print(f"  {r['version']:<8} {r['freq']:<6} {r['oos_sharpe']:>12.2f} {r['oos_return']:>11.2f}% {r['folder']:<30}")
    if fail:
        print(f"\n  Failed:")
        for r in fail:
            print(f"  {r['version']:<8} {r['freq']:<6} {r['status']}")


def main():
    parser = argparse.ArgumentParser(description="Batch optimize new strategies")
    parser.add_argument("--group", choices=list(GROUPS.keys()), help="Run specific group only")
    parser.add_argument("--freq", choices=TIMEFRAMES, help="Run specific frequency only")
    args = parser.parse_args()

    groups = [args.group] if args.group else list(GROUPS.keys())
    for g in groups:
        run_group(g, freq_filter=args.freq)


if __name__ == "__main__":
    main()
