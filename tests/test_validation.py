"""Tests for the 6-layer validation system.

Covers all modules: regime_cv, oos_validator, walk_forward,
deflated_sharpe, monte_carlo, permutation_test, industrial_check,
stress_test, and pipeline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from validation.regime_cv import RegimeCVResult, regime_cv_verdict, run_regime_cv
from validation.oos_validator import OOSResult, validate_oos
from validation.walk_forward import WalkForwardResult, walk_forward_verdict
from validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    sharpe_std_error,
)
from validation.monte_carlo import BootstrapResult, bootstrap_test
from validation.permutation_test import PermutationResult, permutation_test
from validation.industrial_check import IndustrialResult, check_industrial_decay
from validation.stress_test import StressTestResult, check_slippage_sensitivity, run_stress_test
from validation.pipeline import ValidationPipelineResult, run_validation_pipeline


# ============================================================================
# Layer 1: Regime CV
# ============================================================================


class TestRegimeCVVerdict:
    """Test regime_cv_verdict decision logic."""

    def test_pass_high_sharpe_high_winrate(self) -> None:
        assert regime_cv_verdict(0.5, 0.6) == "PASS"

    def test_pass_boundary(self) -> None:
        assert regime_cv_verdict(0.31, 0.5) == "PASS"

    def test_marginal_low_sharpe_ok_winrate(self) -> None:
        assert regime_cv_verdict(0.1, 0.4) == "MARGINAL"

    def test_marginal_boundary(self) -> None:
        assert regime_cv_verdict(0.01, 0.33) == "MARGINAL"

    def test_fail_negative_sharpe(self) -> None:
        assert regime_cv_verdict(-0.5, 0.8) == "FAIL"

    def test_fail_zero_sharpe(self) -> None:
        assert regime_cv_verdict(0.0, 0.5) == "FAIL"

    def test_fail_low_winrate(self) -> None:
        # win_rate 0.3 < 0.33 threshold, so FAIL even with high Sharpe
        assert regime_cv_verdict(0.5, 0.3) == "FAIL"

    def test_fail_both_low(self) -> None:
        assert regime_cv_verdict(0.0, 0.0) == "FAIL"


class TestRunRegimeCV:
    """Test run_regime_cv computation."""

    def test_pass_result(self) -> None:
        result = run_regime_cv([0.5, 0.6, 0.4, 0.3, 0.5], "trend_v1", "strong_trend")
        assert result.verdict == "PASS"
        assert result.n_folds == 5
        assert result.win_rate == 1.0
        assert result.strategy == "trend_v1"
        assert result.regime == "strong_trend"

    def test_fail_all_negative(self) -> None:
        result = run_regime_cv([-0.5, -0.3, -0.8])
        assert result.verdict == "FAIL"
        assert result.win_rate == 0.0
        assert result.mean_sharpe < 0.0

    def test_empty_folds(self) -> None:
        result = run_regime_cv([])
        assert result.verdict == "FAIL"
        assert result.n_folds == 0
        assert result.mean_sharpe == 0.0

    def test_single_fold(self) -> None:
        result = run_regime_cv([0.5])
        assert result.n_folds == 1
        assert result.std_sharpe == 0.0
        assert result.win_rate == 1.0

    def test_marginal_result(self) -> None:
        result = run_regime_cv([0.2, -0.1, 0.1])
        assert result.verdict == "MARGINAL"

    def test_frozen_dataclass(self) -> None:
        result = run_regime_cv([0.5, 0.6])
        with pytest.raises(AttributeError):
            result.verdict = "PASS"  # type: ignore[misc]


# ============================================================================
# Layer 2: OOS Validator
# ============================================================================


class TestOOSValidator:
    """Test OOS validation logic."""

    def test_good_wf_ratio(self) -> None:
        result = validate_oos(is_sharpe=1.0, oos_sharpe=0.8)
        assert result.wf_ratio == pytest.approx(0.8)
        assert "suspected_overfit" not in result.flags

    def test_suspected_overfit(self) -> None:
        result = validate_oos(is_sharpe=2.0, oos_sharpe=0.5)
        assert result.wf_ratio == pytest.approx(0.25)
        assert "suspected_overfit" in result.flags

    def test_behavior_anomaly_trades(self) -> None:
        result = validate_oos(
            is_sharpe=1.0, oos_sharpe=0.8, is_trades=100, oos_trades=20
        )
        assert "behavior_anomaly" in result.flags

    def test_behavior_anomaly_hold_time(self) -> None:
        result = validate_oos(
            is_sharpe=1.0, oos_sharpe=0.8,
            is_avg_hold=10.0, oos_avg_hold=40.0,
        )
        assert "behavior_anomaly" in result.flags

    def test_oos_biased_high(self) -> None:
        result = validate_oos(is_sharpe=1.0, oos_sharpe=2.0)
        assert "oos_biased_high" in result.flags

    def test_no_flags_clean(self) -> None:
        result = validate_oos(is_sharpe=1.0, oos_sharpe=0.9)
        assert len(result.flags) == 0

    def test_zero_is_sharpe_nonzero_oos(self) -> None:
        result = validate_oos(is_sharpe=0.0, oos_sharpe=0.5)
        assert math.isinf(result.wf_ratio)

    def test_zero_is_sharpe_zero_oos(self) -> None:
        result = validate_oos(is_sharpe=0.0, oos_sharpe=0.0)
        assert result.wf_ratio == 0.0

    def test_industrial_decay(self) -> None:
        result = validate_oos(
            is_sharpe=1.0, oos_sharpe=1.0, industrial_sharpe=0.7
        )
        assert result.industrial_decay == pytest.approx(0.3)

    def test_industrial_none(self) -> None:
        result = validate_oos(is_sharpe=1.0, oos_sharpe=0.8)
        assert result.industrial_sharpe is None
        assert result.industrial_decay is None


# ============================================================================
# Layer 3: Walk-Forward
# ============================================================================


class TestWalkForward:
    """Test walk-forward verdict."""

    def test_pass(self) -> None:
        result = walk_forward_verdict([0.5, 0.3, 0.4, 0.6])
        assert result.passed is True
        assert result.win_rate == 1.0
        assert result.mean_sharpe > 0.0

    def test_fail_low_winrate(self) -> None:
        result = walk_forward_verdict([0.5, -0.3, -0.4, -0.6])
        assert result.passed is False

    def test_empty_windows(self) -> None:
        result = walk_forward_verdict([])
        assert result.passed is False
        assert result.n_windows == 0

    def test_worst_best(self) -> None:
        result = walk_forward_verdict([0.1, -0.5, 0.8])
        assert result.worst_sharpe == pytest.approx(-0.5)
        assert result.best_sharpe == pytest.approx(0.8)

    def test_mode_preserved(self) -> None:
        result = walk_forward_verdict([0.5], mode="regime_aware")
        assert result.mode == "regime_aware"

    def test_borderline_win_rate(self) -> None:
        # Exactly 50% win rate, mean > 0
        result = walk_forward_verdict([0.5, -0.1])
        assert result.win_rate == 0.5
        assert result.passed is True


# ============================================================================
# Layer 4: Deflated Sharpe
# ============================================================================


class TestDeflatedSharpe:
    """Test deflated Sharpe ratio functions."""

    def test_expected_max_sharpe_single_trial(self) -> None:
        result = expected_max_sharpe(1, 0.5)
        assert result == 0.0

    def test_expected_max_sharpe_zero_trials(self) -> None:
        result = expected_max_sharpe(0, 0.5)
        assert result == 0.0

    def test_expected_max_sharpe_many_trials(self) -> None:
        result = expected_max_sharpe(100, 0.5)
        assert result > 0.0

    def test_expected_max_sharpe_increases_with_trials(self) -> None:
        e10 = expected_max_sharpe(10, 0.5)
        e100 = expected_max_sharpe(100, 0.5)
        e1000 = expected_max_sharpe(1000, 0.5)
        assert e10 < e100 < e1000

    def test_sharpe_std_error_normal(self) -> None:
        se = sharpe_std_error(1.0, 252)
        assert se > 0.0
        assert se < 1.0

    def test_sharpe_std_error_single_obs(self) -> None:
        se = sharpe_std_error(1.0, 1)
        assert math.isinf(se)

    def test_sharpe_std_error_zero_sharpe(self) -> None:
        se = sharpe_std_error(0.0, 252)
        assert se == pytest.approx(1.0 / math.sqrt(251))

    def test_dsr_high_sharpe_few_trials(self) -> None:
        dsr = deflated_sharpe_ratio(2.0, 5, 0.5, 252)
        assert dsr > 0.5

    def test_dsr_low_sharpe_many_trials(self) -> None:
        dsr = deflated_sharpe_ratio(0.5, 1000, 0.5, 252)
        assert dsr < 0.5

    def test_dsr_returns_probability(self) -> None:
        dsr = deflated_sharpe_ratio(1.5, 50, 0.5, 252)
        assert 0.0 <= dsr <= 1.0

    def test_dsr_with_high_skew(self) -> None:
        dsr_normal = deflated_sharpe_ratio(1.0, 50, 0.5, 252, skew=0.0, kurt=3.0)
        dsr_skewed = deflated_sharpe_ratio(1.0, 50, 0.5, 252, skew=2.0, kurt=3.0)
        # With positive skew and positive sharpe, SE changes
        assert dsr_normal != dsr_skewed


# ============================================================================
# Layer 5a: Bootstrap
# ============================================================================


class TestBootstrap:
    """Test bootstrap resampling."""

    def test_robust_positive_returns(self) -> None:
        rng = np.random.default_rng(123)
        returns = rng.normal(0.001, 0.01, 500)
        result = bootstrap_test(returns, n_sims=500, seed=42)
        assert result.verdict == "ROBUST"
        assert result.sharpe_ci_lower > 0.0

    def test_fragile_zero_mean(self) -> None:
        rng = np.random.default_rng(123)
        returns = rng.normal(0.0, 0.02, 500)
        result = bootstrap_test(returns, n_sims=500, seed=42)
        assert result.verdict in ("FRAGILE", "ACCEPTABLE")

    def test_empty_returns(self) -> None:
        result = bootstrap_test(np.array([]), n_sims=100, seed=42)
        assert result.verdict == "FRAGILE"
        assert result.sharpe_mean == 0.0

    def test_deterministic_with_seed(self) -> None:
        returns = np.random.default_rng(99).normal(0.0005, 0.01, 300)
        r1 = bootstrap_test(returns, n_sims=200, seed=42)
        r2 = bootstrap_test(returns, n_sims=200, seed=42)
        assert r1.sharpe_mean == r2.sharpe_mean
        assert r1.sharpe_ci_lower == r2.sharpe_ci_lower

    def test_maxdd_values(self) -> None:
        returns = np.random.default_rng(55).normal(0.001, 0.01, 300)
        result = bootstrap_test(returns, n_sims=200, seed=42)
        assert result.maxdd_median <= 0.0  # drawdowns are negative
        # 95th percentile is less severe (closer to 0) than median
        assert result.maxdd_95th >= result.maxdd_median

    def test_ci_ordering(self) -> None:
        returns = np.random.default_rng(77).normal(0.0005, 0.01, 300)
        result = bootstrap_test(returns, n_sims=500, seed=42)
        assert result.sharpe_ci_lower <= result.sharpe_mean <= result.sharpe_ci_upper


# ============================================================================
# Layer 5b: Permutation
# ============================================================================


class TestPermutation:
    """Test permutation test."""

    def test_significant_strong_signal(self) -> None:
        # Use a very high Sharpe that random permutations can't match
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 500)
        # Claim a very high sharpe so permutations can't beat it
        result = permutation_test(returns, 5.0, n_perms=500, seed=42)
        assert result.verdict == "SIGNIFICANT"
        assert result.p_value < 0.05

    def test_not_significant_random(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 500)
        # Use a very low sharpe so most permutations beat it
        result = permutation_test(returns, -5.0, n_perms=500, seed=42)
        assert result.verdict == "NOT_SIGNIFICANT"

    def test_empty_returns(self) -> None:
        result = permutation_test(np.array([]), 1.0, n_perms=100, seed=42)
        assert result.verdict == "NOT_SIGNIFICANT"
        assert result.p_value == 1.0

    def test_deterministic_with_seed(self) -> None:
        returns = np.random.default_rng(88).normal(0.001, 0.01, 200)
        r1 = permutation_test(returns, 1.5, n_perms=200, seed=42)
        r2 = permutation_test(returns, 1.5, n_perms=200, seed=42)
        assert r1.p_value == r2.p_value

    def test_p_value_range(self) -> None:
        returns = np.random.default_rng(33).normal(0.0005, 0.01, 300)
        result = permutation_test(returns, 0.5, n_perms=500, seed=42)
        assert 0.0 <= result.p_value <= 1.0


# ============================================================================
# Layer 6a: Industrial Check
# ============================================================================


class TestIndustrialCheck:
    """Test industrial decay check."""

    def test_normal_low_decay(self) -> None:
        result = check_industrial_decay(2.0, 1.85)
        assert result.verdict == "normal"
        assert result.decay_pct < 10.0

    def test_acceptable_decay(self) -> None:
        result = check_industrial_decay(2.0, 1.5)
        assert result.verdict == "acceptable"
        assert 10.0 <= result.decay_pct < 30.0

    def test_warning_decay(self) -> None:
        result = check_industrial_decay(2.0, 1.1)
        assert result.verdict == "warning"
        assert 30.0 <= result.decay_pct < 50.0

    def test_unreliable_high_decay(self) -> None:
        result = check_industrial_decay(2.0, 0.8)
        assert result.verdict == "unreliable"
        assert result.decay_pct >= 50.0

    def test_zero_basic_sharpe_zero_industrial(self) -> None:
        result = check_industrial_decay(0.0, 0.0)
        assert result.decay_pct == 0.0
        assert result.verdict == "normal"

    def test_zero_basic_sharpe_nonzero_industrial(self) -> None:
        result = check_industrial_decay(0.0, 0.5)
        assert result.decay_pct == 100.0
        assert result.verdict == "unreliable"

    def test_frozen_dataclass(self) -> None:
        result = check_industrial_decay(2.0, 1.5)
        with pytest.raises(AttributeError):
            result.verdict = "normal"  # type: ignore[misc]


# ============================================================================
# Layer 6b: Stress Test
# ============================================================================


class TestStressTest:
    """Test stress testing."""

    def test_low_slippage_sensitivity(self) -> None:
        level, decay = check_slippage_sensitivity(2.0, 1.8)
        assert level == "LOW"
        assert decay < 15.0

    def test_moderate_slippage_sensitivity(self) -> None:
        level, decay = check_slippage_sensitivity(2.0, 1.5)
        assert level == "MODERATE"
        assert 15.0 <= decay < 30.0

    def test_high_slippage_sensitivity(self) -> None:
        level, decay = check_slippage_sensitivity(2.0, 1.2)
        assert level == "HIGH"
        assert decay >= 30.0

    def test_run_stress_test_full(self) -> None:
        result = run_stress_test(
            base_sharpe=2.0,
            doubled_slippage_sharpe=1.8,
            cost_doubled_sharpe=1.5,
            adjacent_freq_sharpe=1.2,
            similar_instrument_sharpe=1.0,
        )
        assert result.slippage_sensitivity == "LOW"
        assert result.cost_doubled_sharpe == 1.5
        assert result.adjacent_freq_sharpe == 1.2
        assert result.similar_instrument_sharpe == 1.0

    def test_run_stress_test_optional_none(self) -> None:
        result = run_stress_test(base_sharpe=2.0, doubled_slippage_sharpe=1.0)
        assert result.cost_doubled_sharpe is None
        assert result.adjacent_freq_sharpe is None
        assert result.similar_instrument_sharpe is None

    def test_zero_base_sharpe(self) -> None:
        level, decay = check_slippage_sensitivity(0.0, 0.0)
        assert level == "LOW"
        assert decay == 0.0


# ============================================================================
# Pipeline
# ============================================================================


class TestPipeline:
    """Test the validation pipeline orchestrator."""

    def test_all_none_no_reject(self) -> None:
        result = run_validation_pipeline()
        assert result.hard_reject is False
        assert len(result.reject_reasons) == 0
        assert len(result.soft_flags) == 0

    def test_regime_cv_fail_hard_rejects(self) -> None:
        result = run_validation_pipeline(fold_sharpes=[-1.0, -0.5, -0.3])
        assert result.hard_reject is True
        assert "regime_cv_fail" in result.reject_reasons

    def test_regime_cv_marginal_soft_flag(self) -> None:
        result = run_validation_pipeline(fold_sharpes=[0.1, -0.05, 0.2])
        assert result.hard_reject is False
        assert "regime_cv_marginal" in result.soft_flags

    def test_bootstrap_fragile_hard_rejects(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.001, 0.02, 300)
        result = run_validation_pipeline(
            daily_returns=returns,
            observed_sharpe=-0.5,
            n_bootstrap=200,
            mc_seed=42,
        )
        assert result.hard_reject is True
        assert "bootstrap_fragile" in result.reject_reasons

    def test_industrial_unreliable_hard_rejects(self) -> None:
        result = run_validation_pipeline(
            basic_sharpe=2.0,
            industrial_sharpe=0.5,
            doubled_slippage_sharpe=1.5,
        )
        assert result.hard_reject is True
        assert "industrial_unreliable" in result.reject_reasons

    def test_oos_flags_propagated(self) -> None:
        result = run_validation_pipeline(is_sharpe=2.0, oos_sharpe=0.3)
        assert "oos_suspected_overfit" in result.soft_flags

    def test_walk_forward_unstable_flag(self) -> None:
        result = run_validation_pipeline(window_sharpes=[-0.5, -0.3, 0.1])
        assert "walk_forward_unstable" in result.soft_flags

    def test_deflated_sharpe_low_flag(self) -> None:
        result = run_validation_pipeline(
            observed_sharpe=0.5,
            n_trials=1000,
            sharpe_std=0.5,
            n_obs=252,
        )
        assert "deflated_sharpe_low" in result.soft_flags

    def test_full_pipeline_pass(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 500)
        real_sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252))
        result = run_validation_pipeline(
            fold_sharpes=[0.5, 0.6, 0.4, 0.5],
            strategy="trend_v1",
            regime="strong_trend",
            is_sharpe=1.5,
            oos_sharpe=1.2,
            window_sharpes=[0.5, 0.3, 0.4],
            observed_sharpe=real_sharpe,
            n_trials=5,
            sharpe_std=0.5,
            n_obs=500,
            daily_returns=returns,
            n_bootstrap=200,
            n_permutations=200,
            mc_seed=42,
            basic_sharpe=1.5,
            industrial_sharpe=1.4,
            doubled_slippage_sharpe=1.3,
        )
        assert result.hard_reject is False
        assert result.regime_cv is not None
        assert result.oos is not None
        assert result.walk_forward is not None
        assert result.deflated_sharpe is not None
        assert result.bootstrap is not None
        assert result.permutation is not None
        assert result.industrial is not None
        assert result.stress is not None

    def test_multiple_hard_rejects(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.002, 0.02, 300)
        result = run_validation_pipeline(
            fold_sharpes=[-1.0, -0.5],
            daily_returns=returns,
            observed_sharpe=-1.0,
            n_bootstrap=200,
            mc_seed=42,
            basic_sharpe=2.0,
            industrial_sharpe=0.5,
            doubled_slippage_sharpe=1.0,
        )
        assert result.hard_reject is True
        assert len(result.reject_reasons) >= 2

    def test_pipeline_result_frozen(self) -> None:
        result = run_validation_pipeline()
        with pytest.raises(AttributeError):
            result.hard_reject = True  # type: ignore[misc]

    def test_high_slippage_soft_flag(self) -> None:
        result = run_validation_pipeline(
            basic_sharpe=2.0,
            industrial_sharpe=1.9,
            doubled_slippage_sharpe=1.0,
        )
        assert "high_slippage_sensitivity" in result.soft_flags

    def test_permutation_not_significant_flag(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 300)
        # Use a very low sharpe so permutations easily beat it
        result = run_validation_pipeline(
            daily_returns=returns,
            observed_sharpe=-5.0,
            n_permutations=200,
            mc_seed=42,
        )
        assert "permutation_not_significant" in result.soft_flags
