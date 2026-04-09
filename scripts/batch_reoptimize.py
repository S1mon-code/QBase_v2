#!/usr/bin/env python3
"""Batch re-optimize all daily + 1h strategies for Iron Ore Long / Strong Trend.

Usage:
    python scripts/batch_reoptimize.py
    python scripts/batch_reoptimize.py --freq daily      # daily only
    python scripts/batch_reoptimize.py --freq 1h          # 1h only
    python scripts/batch_reoptimize.py --start v5 --end v10  # range
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

# Paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
from pipeline.qbase_config import ALPHAFORGE_PATH as _AF_PATH, PROJECT_ROOT

for p in (str(_PROJECT_ROOT), _AF_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.dev_pipeline import run_single_strategy_pipeline


def _discover_strategies(base_dir: Path) -> list[tuple[str, str, type]]:
    """Discover all v*.py strategy files in a directory.

    Returns list of (version, module_path, strategy_class).
    """
    strategies = []
    for py_file in sorted(base_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        version = py_file.stem  # e.g. "trend_medium_v1"
        # Extract version number for sorting
        v_num = version.replace("trend_medium_", "")

        # Build module path relative to project root
        rel = py_file.relative_to(_PROJECT_ROOT)
        module_path = str(rel).replace("/", ".").replace(".py", "")

        try:
            mod = importlib.import_module(module_path)
            # Find the strategy class (first QBaseStrategy subclass)
            strategy_cls = None
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (
                    isinstance(obj, type)
                    and hasattr(obj, "regime")
                    and hasattr(obj, "_generate_signal")
                    and obj.__name__ not in (
                        "QBaseStrategy", "TrendingStrategy",
                        "MeanReversionStrategy",
                    )
                ):
                    strategy_cls = obj
                    break

            if strategy_cls is not None:
                strategies.append((v_num, module_path, strategy_cls))
            else:
                print(f"  [skip] {py_file.name}: no strategy class found")
        except Exception as e:
            print(f"  [skip] {py_file.name}: import error: {e}")

    return strategies


def run_batch(
    freq_filter: str | None = None,
    start_v: int | None = None,
    end_v: int | None = None,
):
    """Run batch optimization for daily + 1h strategies."""

    freqs_to_run: dict[str, Path] = {}

    daily_dir = _PROJECT_ROOT / "strategies" / "strong_trend" / "long" / "iron" / "daily"
    h1_dir = _PROJECT_ROOT / "strategies" / "strong_trend" / "long" / "iron" / "1h"

    if freq_filter is None or freq_filter == "daily":
        freqs_to_run["daily"] = daily_dir
    if freq_filter is None or freq_filter == "1h":
        freqs_to_run["1h"] = h1_dir

    results_summary: list[dict] = []
    total_start = time.time()

    for freq, strategy_dir in freqs_to_run.items():
        print(f"\n{'#' * 60}")
        print(f"  BATCH: Iron Ore LONG | Strong Trend | {freq.upper()}")
        print(f"{'#' * 60}")

        strategies = _discover_strategies(strategy_dir)
        print(f"  Found {len(strategies)} strategies in {strategy_dir}")

        for v_name, module_path, strategy_cls in strategies:
            # Extract version number for range filtering
            try:
                v_num = int(v_name.replace("v", "").replace("trend_medium_", ""))
            except ValueError:
                v_num = 0

            if start_v is not None and v_num < start_v:
                continue
            if end_v is not None and v_num > end_v:
                continue

            print(f"\n{'=' * 60}")
            print(f"  >>> {v_name} ({freq}) — {strategy_cls.__name__}")
            print(f"{'=' * 60}")

            t0 = time.time()
            try:
                result = run_single_strategy_pipeline(
                    strategy_class=strategy_cls,
                    symbol="I",
                    direction="long",
                    regime="strong_trend",
                    horizon="medium",
                    version=v_name.replace("trend_medium_", ""),
                    freq=freq,
                )

                elapsed = time.time() - t0
                status = result.get("status", "unknown")
                oos_sharpe = result.get("validation", {}).get("oos_sharpe", None)
                hard_reject = result.get("validation", {}).get("hard_reject", None)

                results_summary.append({
                    "version": v_name,
                    "freq": freq,
                    "status": status,
                    "oos_sharpe": oos_sharpe,
                    "hard_reject": hard_reject,
                    "elapsed_s": round(elapsed, 1),
                })

                print(f"  <<< {v_name} done in {elapsed:.0f}s | status={status} | oos_sharpe={oos_sharpe}")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  <<< {v_name} ERROR in {elapsed:.0f}s: {e}")
                traceback.print_exc()
                results_summary.append({
                    "version": v_name,
                    "freq": freq,
                    "status": "error",
                    "error": str(e),
                    "elapsed_s": round(elapsed, 1),
                })

    # Final summary
    total_elapsed = time.time() - total_start
    print(f"\n\n{'#' * 60}")
    print(f"  BATCH COMPLETE — {len(results_summary)} strategies in {total_elapsed:.0f}s")
    print(f"{'#' * 60}\n")

    # Summary table
    print(f"{'Version':<20} {'Freq':<8} {'Status':<20} {'OOS Sharpe':<12} {'Reject':<8} {'Time':<8}")
    print("-" * 80)
    for r in results_summary:
        oos = f"{r.get('oos_sharpe', 'N/A'):.3f}" if isinstance(r.get("oos_sharpe"), float) else "N/A"
        reject = str(r.get("hard_reject", "N/A"))
        print(f"{r['version']:<20} {r['freq']:<8} {r['status']:<20} {oos:<12} {reject:<8} {r['elapsed_s']:<8.0f}s")

    passed = [r for r in results_summary if r.get("hard_reject") is False]
    failed = [r for r in results_summary if r.get("hard_reject") is True]
    errors = [r for r in results_summary if r.get("status") == "error"]
    gate_fail = [r for r in results_summary if r.get("status") == "failed_bare_gate"]

    print(f"\n  PASSED: {len(passed)}  |  FAILED: {len(failed)}  |  GATE_FAIL: {len(gate_fail)}  |  ERRORS: {len(errors)}")

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch re-optimize strategies")
    parser.add_argument("--freq", choices=["daily", "1h"], default=None, help="Only run this freq")
    parser.add_argument("--start", type=int, default=None, help="Start from version N")
    parser.add_argument("--end", type=int, default=None, help="End at version N")
    args = parser.parse_args()

    run_batch(freq_filter=args.freq, start_v=args.start, end_v=args.end)
