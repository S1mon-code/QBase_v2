#!/usr/bin/env python3
"""Batch optimize ALL 160 new strategies across I and AG, long and short.

Automatically switches regime label files based on direction before optimization.
"""

from __future__ import annotations

import sys
import importlib
import re
import shutil
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AF_PATH = "/Users/simon/Desktop/AlphaForge"

for p in (str(_PROJECT_ROOT), _AF_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.dev_pipeline import run_single_strategy_pipeline


# Groups to optimize
GROUPS = [
    # (regime, direction, instrument, label_file, signal_direction)
    ("strong_trend", "long", "I", "I_long.yaml", "long"),
    ("strong_trend", "long", "AG", "AG_long.yaml", "long"),
    ("strong_trend", "short", "AG", "AG_short.yaml", "short"),
    ("mild_trend", "long", "I", "I_long.yaml", "long"),
    ("mild_trend", "short", "I", "I_short.yaml", "short"),
]

TIMEFRAMES = ["daily", "1h", "2h", "4h"]


def switch_labels(instrument: str, label_file: str):
    """Copy the correct label file to {instrument}.yaml for the optimizer."""
    src = _PROJECT_ROOT / "data" / "regime_labels" / label_file
    dst = _PROJECT_ROOT / "data" / "regime_labels" / f"{instrument}.yaml"
    shutil.copy(src, dst)
    print(f"  Labels: {label_file} → {instrument}.yaml")


def discover_strategies(strategy_dir: Path):
    """Find all v*.py strategy files and return (version, module_path, class)."""
    strategies = []
    for py_file in sorted(strategy_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        v_match = re.search(r"v(\d+)", py_file.stem)
        if not v_match:
            continue
        v_num = v_match.group(0)

        mod_path = str(py_file.relative_to(_PROJECT_ROOT)).replace("/", ".").replace(".py", "")
        if mod_path in sys.modules:
            del sys.modules[mod_path]

        try:
            mod = importlib.import_module(mod_path)
            cls = None
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if (isinstance(obj, type) and hasattr(obj, "regime")
                    and hasattr(obj, "_generate_signal")
                    and obj.__name__ not in ("QBaseStrategy", "TrendingStrategy", "MeanReversionStrategy")):
                    cls = obj
                    break
            if cls is not None:
                strategies.append((v_num, mod_path, cls))
        except Exception as e:
            print(f"    [skip] {py_file.name}: {e}")

    return strategies


def main():
    total_start = time.time()
    all_results = []

    for regime, direction, instrument, label_file, signal_dir in GROUPS:
        print(f"\n{'#' * 60}")
        print(f"  {regime} / {direction} / {instrument}")
        print(f"{'#' * 60}")

        # Switch labels
        switch_labels(instrument, label_file)

        for freq in TIMEFRAMES:
            strategy_dir = _PROJECT_ROOT / "strategies" / regime / direction / instrument / freq
            if not strategy_dir.exists():
                continue

            strategies = discover_strategies(strategy_dir)
            if not strategies:
                continue

            print(f"\n  --- {freq} ({len(strategies)} strategies) ---")

            for v_num, mod_path, cls in strategies:
                t0 = time.time()
                try:
                    result = run_single_strategy_pipeline(
                        strategy_class=cls,
                        symbol=instrument,
                        direction=direction,
                        regime=regime,
                        horizon="medium",
                        version=v_num,
                        freq=freq,
                        params_override={},
                    )
                    elapsed = time.time() - t0
                    st = result.get("status")
                    sr = result.get("validation", {}).get("oos_sharpe")
                    rej = result.get("validation", {}).get("hard_reject")
                    all_results.append((f"{regime}/{direction}/{instrument}/{freq}/{v_num}", st, sr, rej, elapsed))
                    print(f"    {v_num}: {st} | OOS Sharpe={sr if sr else 'N/A'} | {elapsed:.0f}s")
                except Exception as e:
                    elapsed = time.time() - t0
                    all_results.append((f"{regime}/{direction}/{instrument}/{freq}/{v_num}", "error", None, None, elapsed))
                    print(f"    {v_num}: ERROR {str(e)[:60]} | {elapsed:.0f}s")

    # Restore I labels to I_long (default)
    switch_labels("I", "I_long.yaml")

    # Summary
    total = time.time() - total_start
    print(f"\n\n{'=' * 70}")
    print(f"  BATCH COMPLETE — {len(all_results)} strategies in {total:.0f}s")
    print(f"{'=' * 70}")

    passed = [r for r in all_results if r[3] is False]
    failed = [r for r in all_results if r[3] is True]
    gate = [r for r in all_results if r[1] == "failed_bare_gate"]
    errs = [r for r in all_results if r[1] == "error"]

    print(f"  PASSED: {len(passed)} | FAILED: {len(failed)} | GATE_FAIL: {len(gate)} | ERRORS: {len(errs)}")

    # Per-group summary
    for regime, direction, instrument, _, _ in GROUPS:
        group_results = [r for r in all_results if r[0].startswith(f"{regime}/{direction}/{instrument}")]
        group_passed = len([r for r in group_results if r[3] is False])
        group_total = len(group_results)
        print(f"  {regime}/{direction}/{instrument}: {group_passed}/{group_total} passed")


if __name__ == "__main__":
    main()
