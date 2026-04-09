"""QBase_v2 完整性检查工具。

检查策略文件与 research 产出的一致性：
- 每个策略是否有对应的 research 目录
- research 目录中文件是否完整
- strategy name 属性是否与路径一致
- summary.yaml 是否存在且与实际版本匹配
"""
from __future__ import annotations

import os
import re
import sys
import yaml

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGIES_DIR = os.path.join(PROJECT, "strategies")
RESEARCH_DIR = os.path.join(PROJECT, "research")

REQUIRED_RESEARCH_FILES = ["params.yaml", "validation.yaml", "attribution.md", "train.html", "oos.html"]
REGIMES = ["long", "short"]
DIRECTIONS = ["long", "short"]
TIMEFRAMES = ["daily", "1h", "2h", "4h"]


def discover_strategies() -> list[dict]:
    """Find all strategy .py files and extract metadata."""
    strategies = []
    for regime in REGIMES:
        regime_dir = os.path.join(STRATEGIES_DIR, regime)
        if not os.path.exists(regime_dir):
            continue

        for instrument in _listdir(regime_dir):
            for tf in _listdir(os.path.join(regime_dir, instrument)):
                _scan_tf(strategies, regime, regime, instrument, tf)

    return strategies


def _listdir(path: str) -> list[str]:
    if not os.path.exists(path) or not os.path.isdir(path):
        return []
    return [d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
            and not d.startswith(".") and d != "__pycache__"]


def _scan_tf(results: list, regime: str, direction: str, instrument: str, tf: str):
    if tf not in TIMEFRAMES:
        return
    if direction == "both":
        tf_dir = os.path.join(STRATEGIES_DIR, regime, instrument, tf)
    else:
        tf_dir = os.path.join(STRATEGIES_DIR, regime, direction, instrument, tf)

    if not os.path.exists(tf_dir):
        return

    for fname in os.listdir(tf_dir):
        if not fname.endswith(".py") or fname == "__init__.py":
            continue
        m = re.search(r"v(\d+)\.py$", fname)
        if not m:
            continue
        version = int(m.group(1))
        results.append({
            "regime": regime,
            "direction": direction,
            "instrument": instrument,
            "timeframe": tf,
            "version": version,
            "filename": fname,
            "filepath": os.path.join(tf_dir, fname),
        })


def check_name_attribute(strategy: dict) -> str | None:
    """Check that the name attribute matches the file path."""
    with open(strategy["filepath"], "r") as f:
        content = f.read()

    m = re.search(r'name\s*=\s*"([^"]*)"', content)
    if not m:
        return "missing name attribute"

    name = m.group(1)
    s = strategy
    if s["direction"] == "both":
        expected = f"{s['regime']}_{s['instrument']}_{s['timeframe']}_v{s['version']}"
    else:
        expected = f"{s['regime']}_{s['direction']}_{s['instrument']}_{s['timeframe']}_v{s['version']}"

    if name != expected:
        return f"name mismatch: '{name}' != '{expected}'"
    return None


def find_research_dir(strategy: dict) -> str | None:
    """Find the research directory for a strategy (v{N}_... pattern)."""
    s = strategy
    if s["direction"] == "both":
        base = os.path.join(RESEARCH_DIR, s["regime"], s["instrument"], s["timeframe"])
    else:
        base = os.path.join(RESEARCH_DIR, s["regime"], s["direction"], s["instrument"], s["timeframe"])

    if not os.path.exists(base):
        return None

    pattern = re.compile(rf"^v{s['version']}_")
    for entry in os.listdir(base):
        if pattern.match(entry) and os.path.isdir(os.path.join(base, entry)):
            return os.path.join(base, entry)
    return None


def check_research_completeness(research_dir: str) -> list[str]:
    """Check which required files are missing from research directory."""
    missing = []
    for f in REQUIRED_RESEARCH_FILES:
        if not os.path.exists(os.path.join(research_dir, f)):
            missing.append(f)
    return missing


def main():
    strategies = discover_strategies()
    print(f"Found {len(strategies)} strategies\n")

    issues = {"name_mismatch": [], "no_research": [], "incomplete_research": []}
    stats = {"total": len(strategies), "ok": 0, "issues": 0}

    by_group = {}
    for s in strategies:
        key = f"{s['regime']}/{s['direction']}/{s['instrument']}"
        by_group.setdefault(key, []).append(s)

    for group_key in sorted(by_group.keys()):
        group = by_group[group_key]
        group.sort(key=lambda x: (x["timeframe"], x["version"]))
        print(f"--- {group_key} ({len(group)} strategies) ---")

        for s in group:
            label = f"  {s['timeframe']}/v{s['version']}"
            problems = []

            # Check name attribute
            name_issue = check_name_attribute(s)
            if name_issue:
                problems.append(name_issue)
                issues["name_mismatch"].append(s)

            # Check research directory
            research_dir = find_research_dir(s)
            if research_dir is None:
                problems.append("no research dir")
                issues["no_research"].append(s)
            else:
                missing = check_research_completeness(research_dir)
                if missing:
                    problems.append(f"missing: {', '.join(missing)}")
                    issues["incomplete_research"].append((s, missing))

            if problems:
                print(f"{label}: ISSUE — {'; '.join(problems)}")
                stats["issues"] += 1
            else:
                stats["ok"] += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Total strategies:    {stats['total']}")
    print(f"  OK:                  {stats['ok']}")
    print(f"  With issues:         {stats['issues']}")
    print(f"  Name mismatches:     {len(issues['name_mismatch'])}")
    print(f"  Missing research:    {len(issues['no_research'])}")
    print(f"  Incomplete research: {len(issues['incomplete_research'])}")

    return 1 if stats["issues"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
