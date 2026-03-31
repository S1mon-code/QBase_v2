"""6-Layer Validation Pipeline Orchestrator.

Runs all validation layers and aggregates hard rejections and soft flags
into a single ValidationPipelineResult.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from validation.deflated_sharpe import deflated_sharpe_ratio
from validation.industrial_check import IndustrialResult, check_industrial_decay
from validation.monte_carlo import BootstrapResult, bootstrap_test
from validation.oos_validator import OOSResult, validate_oos
from validation.permutation_test import PermutationResult, permutation_test
from validation.regime_cv import RegimeCVResult, run_regime_cv
from validation.stress_test import StressTestResult, run_stress_test
from validation.walk_forward import WalkForwardResult, walk_forward_verdict


@dataclass(frozen=True)
class ValidationPipelineResult:
    """Aggregated result from all 6 validation layers."""

    regime_cv: RegimeCVResult | None
    oos: OOSResult | None
    walk_forward: WalkForwardResult | None
    deflated_sharpe: float | None  # DSR probability
    bootstrap: BootstrapResult | None
    permutation: PermutationResult | None
    industrial: IndustrialResult | None
    stress: StressTestResult | None

    hard_reject: bool  # True if any hard rejection
    reject_reasons: tuple[str, ...]
    soft_flags: tuple[str, ...]


def run_validation_pipeline(
    *,
    # Layer 1: Regime CV
    fold_sharpes: list[float] | None = None,
    strategy: str = "",
    regime: str = "",
    # Layer 2: OOS
    is_sharpe: float | None = None,
    oos_sharpe: float | None = None,
    is_trades: int = 0,
    oos_trades: int = 0,
    is_avg_hold: float = 0.0,
    oos_avg_hold: float = 0.0,
    industrial_sharpe_oos: float | None = None,
    # Layer 3: Walk-Forward
    window_sharpes: list[float] | None = None,
    wf_mode: str = "rolling",
    # Layer 4: Deflated Sharpe
    observed_sharpe: float | None = None,
    n_trials: int | None = None,
    sharpe_std: float | None = None,
    n_obs: int | None = None,
    skew: float = 0.0,
    kurt: float = 3.0,
    # Layer 5: Monte Carlo
    daily_returns: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    mc_seed: int = 42,
    # Layer 6: Industrial + Stress
    basic_sharpe: float | None = None,
    industrial_sharpe: float | None = None,
    doubled_slippage_sharpe: float | None = None,
    cost_doubled_sharpe: float | None = None,
    adjacent_freq_sharpe: float | None = None,
    similar_instrument_sharpe: float | None = None,
) -> ValidationPipelineResult:
    """Run all 6 validation layers. Collects hard rejections and soft flags.

    Each layer is optional: pass None to skip it. Hard rejections:
        - Layer 1: Regime CV verdict == "FAIL"
        - Layer 5a: Bootstrap verdict == "FRAGILE"
        - Layer 6a: Industrial decay > 50% (verdict == "unreliable")

    Soft flags come from all layers' warnings and marginal results.

    Returns:
        ValidationPipelineResult with all layer results and aggregated flags.
    """
    reject_reasons: list[str] = []
    soft_flags: list[str] = []

    # Layer 1: Regime CV
    regime_cv_result: RegimeCVResult | None = None
    if fold_sharpes is not None:
        regime_cv_result = run_regime_cv(fold_sharpes, strategy, regime)
        if regime_cv_result.verdict == "FAIL":
            reject_reasons.append("regime_cv_fail")
        elif regime_cv_result.verdict == "MARGINAL":
            soft_flags.append("regime_cv_marginal")

    # Layer 2: OOS
    oos_result: OOSResult | None = None
    if is_sharpe is not None and oos_sharpe is not None:
        oos_result = validate_oos(
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            is_trades=is_trades,
            oos_trades=oos_trades,
            is_avg_hold=is_avg_hold,
            oos_avg_hold=oos_avg_hold,
            industrial_sharpe=industrial_sharpe_oos,
        )
        for flag in oos_result.flags:
            soft_flags.append(f"oos_{flag}")

    # Layer 3: Walk-Forward
    wf_result: WalkForwardResult | None = None
    if window_sharpes is not None:
        wf_result = walk_forward_verdict(window_sharpes, wf_mode)
        if not wf_result.passed:
            soft_flags.append("walk_forward_unstable")

    # Layer 4: Deflated Sharpe
    dsr_value: float | None = None
    if (
        observed_sharpe is not None
        and n_trials is not None
        and sharpe_std is not None
        and n_obs is not None
    ):
        dsr_value = deflated_sharpe_ratio(
            observed_sharpe, n_trials, sharpe_std, n_obs, skew, kurt
        )
        if dsr_value < 0.95:
            soft_flags.append("deflated_sharpe_low")

    # Layer 5a: Bootstrap
    bootstrap_result: BootstrapResult | None = None
    if daily_returns is not None:
        bootstrap_result = bootstrap_test(daily_returns, n_bootstrap, mc_seed)
        if bootstrap_result.verdict == "FRAGILE":
            reject_reasons.append("bootstrap_fragile")
        elif bootstrap_result.verdict == "ACCEPTABLE":
            soft_flags.append("bootstrap_acceptable")

    # Layer 5b: Permutation
    perm_result: PermutationResult | None = None
    if daily_returns is not None and observed_sharpe is not None:
        perm_result = permutation_test(
            daily_returns, observed_sharpe, n_permutations, mc_seed
        )
        if perm_result.verdict == "NOT_SIGNIFICANT":
            soft_flags.append("permutation_not_significant")
        elif perm_result.verdict == "MARGINAL":
            soft_flags.append("permutation_marginal")

    # Layer 6a: Industrial
    industrial_result: IndustrialResult | None = None
    if basic_sharpe is not None and industrial_sharpe is not None:
        industrial_result = check_industrial_decay(basic_sharpe, industrial_sharpe)
        if industrial_result.verdict == "unreliable":
            reject_reasons.append("industrial_unreliable")
        elif industrial_result.verdict == "warning":
            soft_flags.append("industrial_warning")

    # Layer 6b: Stress
    stress_result: StressTestResult | None = None
    if basic_sharpe is not None and doubled_slippage_sharpe is not None:
        stress_result = run_stress_test(
            base_sharpe=basic_sharpe,
            doubled_slippage_sharpe=doubled_slippage_sharpe,
            cost_doubled_sharpe=cost_doubled_sharpe,
            adjacent_freq_sharpe=adjacent_freq_sharpe,
            similar_instrument_sharpe=similar_instrument_sharpe,
        )
        if stress_result.slippage_sensitivity == "HIGH":
            soft_flags.append("high_slippage_sensitivity")

    return ValidationPipelineResult(
        regime_cv=regime_cv_result,
        oos=oos_result,
        walk_forward=wf_result,
        deflated_sharpe=dsr_value,
        bootstrap=bootstrap_result,
        permutation=perm_result,
        industrial=industrial_result,
        stress=stress_result,
        hard_reject=len(reject_reasons) > 0,
        reject_reasons=tuple(reject_reasons),
        soft_flags=tuple(soft_flags),
    )
