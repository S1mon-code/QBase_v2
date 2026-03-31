"""Tests for pipeline.runner and pipeline.cli.

Covers:
- PipelineConfig defaults and immutability
- PipelineResult.success property
- QBasePipeline step execution, ordering, skip/error handling
- CLI parser: all subcommands, required args, defaults
- CLI main: dispatch, help, exit codes
"""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml

from pipeline.runner import (
    PIPELINE_STEPS,
    PipelineConfig,
    PipelineResult,
    QBasePipeline,
)
from pipeline.cli import create_parser, main


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def regime_labels_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary regime_labels directory and patch the matcher."""
    labels_dir = tmp_path / "data" / "regime_labels"
    labels_dir.mkdir(parents=True)

    # Patch the matcher's _LABELS_DIR to point to our temp dir
    import regime.matcher as _matcher_mod

    monkeypatch.setattr(_matcher_mod, "_LABELS_DIR", labels_dir)
    return labels_dir


def _write_label_file(labels_dir: Path, instrument: str, labels: list[dict]) -> Path:
    """Helper: write a YAML label file into *labels_dir*."""
    data = {
        "instrument": instrument,
        "version": 1,
        "labeled_by": "test",
        "labels": labels,
    }
    path = labels_dir / f"{instrument}.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return path


# ======================================================================
# PipelineConfig
# ======================================================================


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig(instrument="RB")
        assert cfg.instrument == "RB"
        assert cfg.freq == "1h"
        assert cfg.regime == "strong_trend"
        assert cfg.direction == "up"
        assert cfg.n_trials == 80
        assert cfg.industrial is True

    def test_custom_values(self) -> None:
        cfg = PipelineConfig(
            instrument="I",
            freq="4h",
            regime="mean_reversion",
            direction="down",
            n_trials=120,
            industrial=False,
        )
        assert cfg.instrument == "I"
        assert cfg.freq == "4h"
        assert cfg.regime == "mean_reversion"
        assert cfg.direction == "down"
        assert cfg.n_trials == 120
        assert cfg.industrial is False

    def test_immutable(self) -> None:
        cfg = PipelineConfig(instrument="RB")
        with pytest.raises(AttributeError):
            cfg.instrument = "I"  # type: ignore[misc]


# ======================================================================
# PipelineResult
# ======================================================================


class TestPipelineResult:
    def test_success_when_no_errors(self) -> None:
        result = PipelineResult(steps_completed=["label"])
        assert result.success is True

    def test_failure_when_errors_present(self) -> None:
        result = PipelineResult(errors={"label": "boom"})
        assert result.success is False

    def test_empty_result_is_success(self) -> None:
        result = PipelineResult()
        assert result.success is True

    def test_skipped_does_not_affect_success(self) -> None:
        result = PipelineResult(
            steps_completed=["label"],
            steps_skipped=["optimize"],
        )
        assert result.success is True


# ======================================================================
# QBasePipeline
# ======================================================================


class TestQBasePipeline:
    def test_run_all_no_labels_file(self, regime_labels_dir: Path) -> None:
        """When no label file exists, label step errors."""
        cfg = PipelineConfig(instrument="MISSING")
        pipeline = QBasePipeline(cfg)
        result = pipeline.run_all()

        assert "label" in result.errors
        assert "MISSING" in result.errors["label"]
        # Remaining steps should still be attempted
        assert len(result.steps_skipped) + len(result.steps_completed) + len(result.errors) == len(
            PIPELINE_STEPS
        )

    def test_run_all_empty_labels(self, regime_labels_dir: Path) -> None:
        """When label file exists but has no matching periods, label errors."""
        _write_label_file(regime_labels_dir, "RB", [])
        cfg = PipelineConfig(instrument="RB")
        pipeline = QBasePipeline(cfg)
        result = pipeline.run_all()

        assert "label" in result.errors

    def test_run_all_valid_labels(self, regime_labels_dir: Path) -> None:
        """Label step completes; remaining steps are skipped (NotImplemented)."""
        _write_label_file(
            regime_labels_dir,
            "RB",
            [
                {
                    "start": "2020-01-01",
                    "end": "2020-06-30",
                    "regime": "strong_trend",
                    "direction": "up",
                    "split": "train",
                }
            ],
        )
        cfg = PipelineConfig(instrument="RB")
        pipeline = QBasePipeline(cfg)
        result = pipeline.run_all()

        assert "label" in result.steps_completed
        assert "optimize" in result.steps_skipped
        assert "validate" in result.steps_skipped
        assert "attribute" in result.steps_skipped
        assert "portfolio" in result.steps_skipped
        assert result.success is True

    def test_step_ordering(self, regime_labels_dir: Path) -> None:
        """Steps are attempted in the canonical order."""
        _write_label_file(
            regime_labels_dir,
            "I",
            [
                {
                    "start": "2019-03-01",
                    "end": "2019-09-30",
                    "regime": "strong_trend",
                    "direction": "up",
                    "split": "train",
                }
            ],
        )
        cfg = PipelineConfig(instrument="I")
        pipeline = QBasePipeline(cfg)
        result = pipeline.run_all()

        all_steps = result.steps_completed + result.steps_skipped
        assert all_steps == list(PIPELINE_STEPS)

    def test_run_step_valid(self, regime_labels_dir: Path) -> None:
        """run_step with a recognised step name works."""
        _write_label_file(
            regime_labels_dir,
            "RB",
            [
                {
                    "start": "2020-01-01",
                    "end": "2020-06-30",
                    "regime": "strong_trend",
                    "direction": "up",
                    "split": "train",
                }
            ],
        )
        cfg = PipelineConfig(instrument="RB")
        pipeline = QBasePipeline(cfg)
        result = pipeline.run_step("label")

        assert result.steps_completed == ["label"]
        assert result.success is True

    def test_run_step_invalid_name(self) -> None:
        cfg = PipelineConfig(instrument="RB")
        pipeline = QBasePipeline(cfg)
        with pytest.raises(ValueError, match="Unknown step"):
            pipeline.run_step("nonexistent")

    def test_run_step_not_implemented(self) -> None:
        """Not-implemented steps are recorded as skipped, not errors."""
        cfg = PipelineConfig(instrument="RB")
        pipeline = QBasePipeline(cfg)
        result = pipeline.run_step("optimize")

        assert result.steps_skipped == ["optimize"]
        assert result.success is True

    def test_errors_do_not_block_subsequent_steps(self, regime_labels_dir: Path) -> None:
        """An error in label does not prevent later steps from running."""
        cfg = PipelineConfig(instrument="NOPE")
        pipeline = QBasePipeline(cfg)
        result = pipeline.run_all()

        assert "label" in result.errors
        # At least some subsequent steps attempted (skipped as NotImplemented)
        assert len(result.steps_skipped) > 0

    def test_result_reset_on_rerun(self, regime_labels_dir: Path) -> None:
        """Calling run_all twice produces a fresh result each time."""
        _write_label_file(
            regime_labels_dir,
            "HC",
            [
                {
                    "start": "2021-01-01",
                    "end": "2021-12-31",
                    "regime": "strong_trend",
                    "direction": "up",
                    "split": "train",
                }
            ],
        )
        cfg = PipelineConfig(instrument="HC")
        pipeline = QBasePipeline(cfg)

        r1 = pipeline.run_all()
        r2 = pipeline.run_all()

        assert r1.steps_completed == r2.steps_completed
        assert r1.steps_skipped == r2.steps_skipped


# ======================================================================
# CLI Parser
# ======================================================================


class TestCLIParser:
    def test_label_subcommand(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["label", "I", "--visualize"])
        assert args.command == "label"
        assert args.instrument == "I"
        assert args.visualize is True

    def test_label_validate_flag(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["label", "RB", "--validate"])
        assert args.validate is True

    def test_run_subcommand(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["run", "trend_medium_v1", "--symbol", "RB"])
        assert args.command == "run"
        assert args.strategy == "trend_medium_v1"
        assert args.symbol == "RB"
        assert args.freq == "1h"

    def test_optimize_subcommand(self) -> None:
        parser = create_parser()
        args = parser.parse_args([
            "optimize", "trend_medium_v1", "--symbol", "RB",
            "--trials", "120", "--multi-seed",
        ])
        assert args.command == "optimize"
        assert args.trials == 120
        assert args.multi_seed is True

    def test_optimize_defaults(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["optimize", "s1", "--symbol", "I"])
        assert args.regime == "strong_trend"
        assert args.direction == "up"
        assert args.trials == 80

    def test_validate_all_flag(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["validate", "trend_medium_v1", "--all"])
        assert args.all is True

    def test_validate_individual_layers(self) -> None:
        parser = create_parser()
        args = parser.parse_args([
            "validate", "s1", "--regime-cv", "--oos", "--walk-forward",
            "--dsr", "--monte-carlo", "--industrial",
        ])
        assert args.regime_cv is True
        assert args.oos is True
        assert args.walk_forward is True
        assert args.dsr is True
        assert args.monte_carlo is True
        assert args.industrial is True

    def test_attribute_subcommand(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["attribute", "trend_medium_v1", "--symbol", "RB"])
        assert args.command == "attribute"
        assert args.symbol == "RB"

    def test_portfolio_build(self) -> None:
        parser = create_parser()
        args = parser.parse_args([
            "portfolio", "build", "--symbol", "RB", "--regime", "strong_trend",
        ])
        assert args.command == "portfolio"
        assert args.portfolio_action == "build"
        assert args.symbol == "RB"
        assert args.regime == "strong_trend"

    def test_portfolio_score(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["portfolio", "score", "--symbol", "HC"])
        assert args.portfolio_action == "score"
        assert args.symbol == "HC"

    def test_pipeline_subcommand(self) -> None:
        parser = create_parser()
        args = parser.parse_args([
            "pipeline", "--symbol", "RB", "--regime", "mild_trend",
            "--direction", "down", "--trials", "50",
        ])
        assert args.command == "pipeline"
        assert args.symbol == "RB"
        assert args.regime == "mild_trend"
        assert args.direction == "down"
        assert args.trials == 50

    def test_pipeline_defaults(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["pipeline", "--symbol", "RB"])
        assert args.freq == "1h"
        assert args.regime == "strong_trend"
        assert args.direction == "up"
        assert args.trials == 80

    def test_missing_required_arg_raises(self) -> None:
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["optimize", "s1"])  # --symbol missing


# ======================================================================
# CLI main()
# ======================================================================


class TestCLIMain:
    def test_no_command_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main([])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "qbase" in captured.out.lower()

    def test_pipeline_command_with_valid_labels(
        self, regime_labels_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_label_file(
            regime_labels_dir,
            "RB",
            [
                {
                    "start": "2020-01-01",
                    "end": "2020-06-30",
                    "regime": "strong_trend",
                    "direction": "up",
                    "split": "train",
                }
            ],
        )
        exit_code = main(["pipeline", "--symbol", "RB"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "label" in captured.out

    def test_pipeline_command_with_missing_labels(
        self, regime_labels_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["pipeline", "--symbol", "NOPE"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Errors" in captured.out

    def test_placeholder_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Commands not yet connected to AlphaForge return a placeholder."""
        exit_code = main(["optimize", "s1", "--symbol", "RB"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "AlphaForge" in captured.out

    def test_run_command_placeholder(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["run", "trend_medium_v1", "--symbol", "I"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "AlphaForge" in captured.out

    def test_label_command_missing_file(
        self, regime_labels_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["label", "UNKNOWN"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out
