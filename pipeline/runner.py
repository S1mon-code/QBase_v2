"""QBase_v2 full pipeline orchestrator.

Orchestrates the flow: label -> optimize -> validate -> attribute -> portfolio.
Each step can be run independently or as part of the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


PIPELINE_STEPS = ("label", "optimize", "validate", "attribute", "portfolio")


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for a single pipeline run.

    Attributes:
        instrument: Futures instrument code (e.g. 'RB', 'I').
        freq: Bar frequency.
        regime: Regime type to target.
        direction: Directional constraint.
        n_trials: Number of optimization trials.
        industrial: Whether to use AlphaForge Industrial mode.
    """

    instrument: str
    freq: str = "1h"
    regime: str = "strong_trend"
    direction: str = "up"
    n_trials: int = 80
    industrial: bool = True


@dataclass
class PipelineResult:
    """Accumulated result from a pipeline run.

    Attributes:
        steps_completed: Steps that finished successfully.
        steps_skipped: Steps skipped (e.g. not yet implemented).
        errors: Mapping of step name to error message for failures.
    """

    steps_completed: list[str] = field(default_factory=list)
    steps_skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True when no step produced an error."""
        return len(self.errors) == 0


class QBasePipeline:
    """Full pipeline orchestrator.

    Steps:
      1. label     - Verify regime labels exist for the instrument.
      2. optimize  - Run optimizer on train split.
      3. validate  - Run 6-layer validation.
      4. attribute - Run 5-layer attribution.
      5. portfolio - Build portfolio with signal blender.

    Each step checks prerequisites and records skips when the
    AlphaForge backtest connection is not yet available.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.result = PipelineResult()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> PipelineResult:
        """Run every pipeline step in order.

        Returns a new ``PipelineResult`` summarising outcomes.  Steps that
        depend on an AlphaForge connection log a skip rather than an error.
        """
        self.result = PipelineResult()
        for step in PIPELINE_STEPS:
            self._run_step(step)
        return self.result

    def run_step(self, step: str) -> PipelineResult:
        """Run a single named step.

        Args:
            step: One of the values in ``PIPELINE_STEPS``.

        Raises:
            ValueError: If *step* is not a recognised pipeline step.
        """
        if step not in PIPELINE_STEPS:
            raise ValueError(
                f"Unknown step '{step}'. Valid steps: {PIPELINE_STEPS}"
            )
        self.result = PipelineResult()
        self._run_step(step)
        return self.result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_step(self, step: str) -> None:
        handler = getattr(self, f"_step_{step}")
        try:
            handler()
            self.result.steps_completed.append(step)
        except NotImplementedError:
            self.result.steps_skipped.append(step)
        except Exception as exc:
            self.result.errors[step] = str(exc)

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _step_label(self) -> None:
        """Check regime labels exist for the configured instrument."""
        from regime.matcher import get_regime_periods

        periods = get_regime_periods(
            self.config.instrument,
            self.config.regime,
            self.config.direction,
            "train",
        )
        if not periods:
            raise ValueError(
                f"No {self.config.regime}/{self.config.direction} train "
                f"periods for {self.config.instrument}"
            )

    def _step_optimize(self) -> None:
        """Run optimisation.  Requires AlphaForge connection."""
        # NOTE: See AlphaForge CLAUDE.md for backtesting API
        raise NotImplementedError(
            "Connect to AlphaForge V6.0 for optimization"
        )

    def _step_validate(self) -> None:
        """Run validation.  Requires optimisation results."""
        raise NotImplementedError(
            "Connect to AlphaForge V6.0 for validation"
        )

    def _step_attribute(self) -> None:
        """Run attribution.  Requires validated strategies."""
        raise NotImplementedError(
            "Connect to AlphaForge V6.0 for attribution"
        )

    def _step_portfolio(self) -> None:
        """Build portfolio.  Requires attributed strategies."""
        raise NotImplementedError(
            "Connect to AlphaForge V6.0 for portfolio construction"
        )
