"""Run portfolio construction pipeline.

Usage:
    PYTHONPATH=.:../AlphaForge python scripts/run_portfolio.py
    PYTHONPATH=.:../AlphaForge python scripts/run_portfolio.py --method risk_parity
    PYTHONPATH=.:../AlphaForge python scripts/run_portfolio.py --top-n 5 --corr-threshold 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure scripts/ is on path for sibling imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Ensure project root is on path for pipeline imports
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_engine import build_portfolio  # noqa: E402
from sqs import scan_all_strategies, apply_kill_switch  # noqa: E402

RESEARCH_DIR = PROJECT_ROOT / "research"
REPORT_DIR = PROJECT_ROOT / "reports" / "portfolio"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt_pct(val: float) -> str:
    return f"{val * 100:+.2f}%"


def _print_sqs_table(strategies: list[dict], title: str, limit: int = 20) -> None:
    """Print a formatted table of strategies with SQS scores."""
    print(f"\n{'=' * 90}")
    print(f"  {title}  ({len(strategies)} total, showing top {min(limit, len(strategies))})")
    print(f"{'=' * 90}")
    header = (
        f"{'#':>3}  {'Quadrant':<14} {'Freq':<6} {'Version':<20} "
        f"{'SQS':>5} {'Sharpe':>7} {'Return':>9} {'Trades':>6} {'Kill':>5}"
    )
    print(header)
    print("-" * 90)

    for i, s in enumerate(strategies[:limit], 1):
        quadrant = s.get("quadrant", "?")
        freq = s.get("freq", "?")
        version = s.get("version_dir", "?")
        sqs = s.get("sqs", 0.0)
        sharpe = s.get("oos_sharpe", 0.0)
        ret = s.get("oos_return", 0.0)
        trades = s.get("oos_n_trades", 0)
        kill = "KILL" if s.get("kill", False) else ""

        # Truncate version if too long
        if len(version) > 18:
            version = version[:18] + ".."

        print(
            f"{i:>3}  {quadrant:<14} {freq:<6} {version:<20} "
            f"{sqs:>5.1f} {sharpe:>7.3f} {_fmt_pct(ret):>9} {trades:>6} {kill:>5}"
        )
    print()


def _print_kill_summary(killed: list[dict]) -> None:
    """Print kill reasons summary."""
    from collections import Counter

    reason_counts: Counter[str] = Counter()
    for s in killed:
        for r in s.get("kill_reasons", []):
            reason_counts[r] += 1

    print(f"\n{'=' * 50}")
    print(f"  Kill Reasons Summary ({len(killed)} killed)")
    print(f"{'=' * 50}")
    for reason, count in reason_counts.most_common():
        print(f"  {reason:<35} {count:>4}")
    print()


def _print_portfolio(portfolio: dict) -> None:
    """Print portfolio summary."""
    metrics = portfolio["metrics"]
    strategies = portfolio["strategies"]

    print(f"\n{'=' * 90}")
    print(f"  PORTFOLIO SUMMARY  (method: {portfolio['method']})")
    print(f"{'=' * 90}")
    print(f"  Scanned: {portfolio['total_scanned']}")
    print(f"  Killed:  {portfolio['total_killed']}")
    print(f"  Candidates: {portfolio['total_candidates']}")
    print(f"  Selected: {portfolio['total_selected']}")
    print(f"  Instruments: {', '.join(metrics.get('instruments', []))}")
    print(f"  Frequencies: {', '.join(metrics.get('freqs', []))}")
    print(f"  Directions:  {', '.join(metrics.get('directions', []))}")
    print(f"  Avg SQS:     {metrics['avg_sqs']:.1f}")
    print(f"  Exp Sharpe:  {metrics['expected_sharpe']:.4f}")
    print(f"  Exp Return:  {_fmt_pct(metrics['expected_return'])}")
    print(f"  Total Lots:  {metrics['total_lots']}")
    print()

    if portfolio.get("diversification_warnings"):
        print("  WARNINGS:")
        for w in portfolio["diversification_warnings"]:
            print(f"    - {w}")
        print()

    # Strategy detail table
    print(f"  {'#':>3}  {'Quadrant':<14} {'Freq':<6} {'Version':<20} "
          f"{'SQS':>5} {'Weight':>7} {'Lots':>5} {'Sharpe':>7} {'Return':>9}")
    print("  " + "-" * 86)

    for i, s in enumerate(strategies, 1):
        quadrant = s.get("quadrant", "?")
        freq = s.get("freq", "?")
        version = s.get("version_dir", "?")
        if len(version) > 18:
            version = version[:18] + ".."
        sqs = s.get("sqs", 0.0)
        weight = s.get("weight", 0.0)
        lots = s.get("lots", 0)
        sharpe = s.get("oos_sharpe", 0.0)
        ret = s.get("oos_return", 0.0)

        print(
            f"  {i:>3}  {quadrant:<14} {freq:<6} {version:<20} "
            f"{sqs:>5.1f} {weight:>6.1%} {lots:>5} {sharpe:>7.3f} {_fmt_pct(ret):>9}"
        )
    print()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save_portfolio(portfolio: dict, output_path: Path) -> None:
    """Save portfolio results to YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Simplify for YAML output (remove large nested dicts)
    summary = {
        "config": portfolio["config"],
        "method": portfolio["method"],
        "counts": {
            "scanned": portfolio["total_scanned"],
            "killed": portfolio["total_killed"],
            "survivors": portfolio["total_survivors"],
            "candidates": portfolio["total_candidates"],
            "selected": portfolio["total_selected"],
        },
        "metrics": portfolio["metrics"],
        "diversification_warnings": portfolio["diversification_warnings"],
        "strategies": [
            {
                "quadrant": s["quadrant"],
                "direction": s["direction"],
                "instrument": s["instrument"],
                "freq": s["freq"],
                "version_dir": s["version_dir"],
                "sqs": s["sqs"],
                "weight": s.get("weight", 0.0),
                "lots": s.get("lots", 0),
                "oos_sharpe": s["oos_sharpe"],
                "oos_return": s["oos_return"],
                "oos_max_dd": s["oos_max_dd"],
                "dimensions": s["dimensions"],
            }
            for s in portfolio["strategies"]
        ],
    }

    with open(output_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)

    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QBase_v2 Portfolio Builder")
    parser.add_argument(
        "--method",
        choices=["equal", "risk_parity", "sqs_weighted"],
        default="sqs_weighted",
        help="Weighting method (default: sqs_weighted)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Top N strategies per quadrant (default: 3)",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.7,
        help="Correlation threshold for dedup (default: 0.7)",
    )
    parser.add_argument(
        "--min-sqs",
        type=float,
        default=30,
        help="Minimum SQS for candidate selection (default: 30)",
    )
    parser.add_argument(
        "--max-per-instrument",
        type=int,
        default=3,
        help="Max strategies per instrument (default: 3)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all strategies (not just top 20)",
    )
    args = parser.parse_args()

    config = {
        "top_n_per_quadrant": args.top_n,
        "corr_threshold": args.corr_threshold,
        "min_sqs": args.min_sqs,
        "max_per_instrument": args.max_per_instrument,
    }

    print("QBase_v2 Portfolio Builder")
    print(f"Research dir: {RESEARCH_DIR}")
    print()

    # Step 1: Scan all strategies
    print("Scanning strategies...")
    all_strategies = scan_all_strategies(RESEARCH_DIR)
    print(f"  Found {len(all_strategies)} strategies")

    # Step 2: Show SQS summary
    display_limit = len(all_strategies) if args.show_all else 20
    _print_sqs_table(all_strategies, "All Strategies by SQS", limit=display_limit)

    # Step 3: Kill switch
    survivors, killed = apply_kill_switch(all_strategies)
    print(f"Survivors: {len(survivors)}, Killed: {len(killed)}")
    _print_kill_summary(killed)

    # Step 4: Build portfolio
    print("Building portfolio...")
    portfolio = build_portfolio(
        RESEARCH_DIR,
        method=args.method,
        config=config,
    )

    # Step 5: Display portfolio
    _print_portfolio(portfolio)

    # Step 6: Save
    output_path = REPORT_DIR / "portfolio_summary.yaml"
    _save_portfolio(portfolio, output_path)


if __name__ == "__main__":
    main()
