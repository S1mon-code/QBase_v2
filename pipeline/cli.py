"""QBase_v2 command-line interface.

Usage examples::

    python -m pipeline.cli label I --visualize
    python -m pipeline.cli run trend_medium_v1 --symbol RB --freq 1h
    python -m pipeline.cli optimize trend_medium_v1 --symbol RB --regime strong_trend
    python -m pipeline.cli validate trend_medium_v1 --all
    python -m pipeline.cli attribute trend_medium_v1 --symbol RB
    python -m pipeline.cli portfolio build --symbol RB --regime strong_trend
    python -m pipeline.cli pipeline --symbol RB --regime strong_trend --direction up
"""

from __future__ import annotations

import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands."""

    parser = argparse.ArgumentParser(
        prog="qbase",
        description="QBase_v2 -- Black series futures multi-strategy system",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- label --------------------------------------------------------
    label_parser = subparsers.add_parser("label", help="Regime labeling")
    label_parser.add_argument("instrument", help="Instrument code (e.g. RB, I)")
    label_parser.add_argument("--visualize", action="store_true", help="Generate overlay chart")
    label_parser.add_argument("--validate", action="store_true", help="Check label completeness")

    # ---- run ----------------------------------------------------------
    run_parser = subparsers.add_parser("run", help="Run a strategy backtest")
    run_parser.add_argument("strategy", help="Strategy file or name")
    run_parser.add_argument("--symbol", required=True)
    run_parser.add_argument("--freq", default="1h")
    run_parser.add_argument("--start", type=int, default=None, help="Start year")
    run_parser.add_argument("--regime", default=None)
    run_parser.add_argument("--direction", default=None)

    # ---- optimize -----------------------------------------------------
    opt_parser = subparsers.add_parser("optimize", help="Strategy optimization")
    opt_parser.add_argument("strategy", help="Strategy file or name")
    opt_parser.add_argument("--symbol", required=True)
    opt_parser.add_argument("--freq", default="1h")
    opt_parser.add_argument("--regime", default="strong_trend")
    opt_parser.add_argument("--direction", default="up")
    opt_parser.add_argument("--trials", type=int, default=80)
    opt_parser.add_argument("--multi-seed", action="store_true")

    # ---- validate -----------------------------------------------------
    val_parser = subparsers.add_parser("validate", help="Strategy validation")
    val_parser.add_argument("strategy", help="Strategy name")
    val_parser.add_argument("--regime-cv", action="store_true")
    val_parser.add_argument("--oos", action="store_true")
    val_parser.add_argument("--walk-forward", action="store_true")
    val_parser.add_argument("--dsr", action="store_true")
    val_parser.add_argument("--monte-carlo", action="store_true")
    val_parser.add_argument("--industrial", action="store_true")
    val_parser.add_argument("--all", action="store_true", help="Run all 6 validation layers")

    # ---- attribute ----------------------------------------------------
    attr_parser = subparsers.add_parser("attribute", help="Attribution analysis")
    attr_parser.add_argument("strategy", help="Strategy name")
    attr_parser.add_argument("--symbol", required=True)

    # ---- portfolio ----------------------------------------------------
    port_parser = subparsers.add_parser("portfolio", help="Portfolio construction")
    port_sub = port_parser.add_subparsers(dest="portfolio_action")

    build_parser = port_sub.add_parser("build", help="Build portfolio")
    build_parser.add_argument("--symbol", required=True)
    build_parser.add_argument("--regime", required=True)

    score_parser = port_sub.add_parser("score", help="Score portfolio")
    score_parser.add_argument("--symbol", required=True)

    report_parser = port_sub.add_parser("report", help="Generate portfolio report")
    report_parser.add_argument("--symbol", required=True)

    # ---- pipeline (full run) ------------------------------------------
    pipe_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipe_parser.add_argument("--symbol", required=True)
    pipe_parser.add_argument("--freq", default="1h")
    pipe_parser.add_argument("--regime", default="strong_trend")
    pipe_parser.add_argument("--direction", default="up")
    pipe_parser.add_argument("--trials", type=int, default=80)

    return parser


def _handle_pipeline(args: argparse.Namespace) -> int:
    """Execute the full pipeline orchestrator."""
    from pipeline.runner import PipelineConfig, QBasePipeline

    config = PipelineConfig(
        instrument=args.symbol,
        freq=args.freq,
        regime=args.regime,
        direction=args.direction,
        n_trials=args.trials,
    )
    pipeline = QBasePipeline(config)
    result = pipeline.run_all()

    print(f"Completed: {result.steps_completed}")
    print(f"Skipped:   {result.steps_skipped}")
    if result.errors:
        print(f"Errors:    {result.errors}")
    return 0 if result.success else 1


def _handle_label(args: argparse.Namespace) -> int:
    """Handle the label subcommand."""
    from regime.matcher import get_regime_periods

    try:
        periods = get_regime_periods(args.instrument, "strong_trend")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Found {len(periods)} regime periods for {args.instrument}")
    if args.validate:
        from regime.schema import load_labels, validate_labels
        from pathlib import Path

        path = Path("data/regime_labels") / f"{args.instrument}.yaml"
        config = load_labels(path)
        errors = validate_labels(config)
        if errors:
            for err in errors:
                print(f"  WARN: {err}")
            return 1
        print("  Labels valid.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Argument list.  Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 = success).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "pipeline":
        return _handle_pipeline(args)

    if args.command == "label":
        return _handle_label(args)

    # Commands that require AlphaForge connection
    print(
        f"Command '{args.command}' received. "
        "Connect to AlphaForge V6.0 to execute."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
