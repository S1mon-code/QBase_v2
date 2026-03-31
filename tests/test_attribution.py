"""Tests for attribution module -- 5-layer attribution analysis.

Covers: signal, horizon, regime, baseline, operational, coverage, decay, report.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from attribution.signal import (
    SignalAttributionResult,
    shapley_attribution,
    ablation_attribution,
    auto_attribution,
)
from attribution.horizon import HorizonAttributionResult, horizon_attribution
from attribution.regime import RegimeStats, RegimeAttributionResult, regime_attribution
from attribution.baseline import BaselineDecomposition, decompose_baseline
from attribution.operational import OperationalAttribution, operational_attribution
from attribution.coverage import CoverageResult, regime_coverage
from attribution.decay import DecayResult, detect_alpha_decay
from attribution.report import generate_attribution_report


# ---------------------------------------------------------------------------
# Layer A: Signal Attribution
# ---------------------------------------------------------------------------

class TestShapleyAttribution:
    """Tests for Shapley value signal attribution."""

    def test_two_signals_equal_contribution(self):
        """Two equally contributing signals should have equal Shapley values."""
        def evaluate(active: set[str]) -> float:
            return len(active) * 0.5

        result = shapley_attribution(["A", "B"], evaluate)
        assert result.method == "shapley"
        assert abs(result.baseline_sharpe - 1.0) < 1e-6
        for c in result.contributions:
            assert abs(c.contribution - 0.5) < 1e-6

    def test_two_signals_one_dominant(self):
        """One signal contributes all Sharpe; other is redundant."""
        def evaluate(active: set[str]) -> float:
            return 2.0 if "A" in active else 0.0

        result = shapley_attribution(["A", "B"], evaluate)
        assert result.dominant == "A"
        assert "B" in result.redundant

    def test_three_signals(self):
        """Shapley values for 3 signals sum to baseline Sharpe."""
        def evaluate(active: set[str]) -> float:
            val = 0.0
            if "A" in active:
                val += 1.0
            if "B" in active:
                val += 0.5
            if "C" in active:
                val += 0.3
            return val

        result = shapley_attribution(["A", "B", "C"], evaluate)
        total = sum(c.contribution for c in result.contributions)
        assert abs(total - result.baseline_sharpe) < 1e-6

    def test_four_signals_sum_to_baseline(self):
        """Shapley values for 4 signals sum to baseline Sharpe."""
        def evaluate(active: set[str]) -> float:
            return sum({"W": 0.4, "X": 0.3, "Y": 0.2, "Z": 0.1}.get(s, 0) for s in active)

        result = shapley_attribution(["W", "X", "Y", "Z"], evaluate)
        total = sum(c.contribution for c in result.contributions)
        assert abs(total - 1.0) < 1e-6

    def test_empty_signals(self):
        """Empty signal list returns empty result."""
        result = shapley_attribution([], lambda x: 0.0)
        assert result.contributions == ()
        assert result.dominant == ""

    def test_redundant_detection_below_5pct(self):
        """Signals contributing <5% are flagged as redundant."""
        def evaluate(active: set[str]) -> float:
            val = 0.0
            if "A" in active:
                val += 2.0
            if "B" in active:
                val += 0.05  # ~2.4% of 2.05
            return val

        result = shapley_attribution(["A", "B"], evaluate)
        assert "B" in result.redundant

    def test_contributions_sorted_descending(self):
        """Contributions are sorted by value descending."""
        def evaluate(active: set[str]) -> float:
            return sum({"A": 0.3, "B": 0.7}.get(s, 0) for s in active)

        result = shapley_attribution(["A", "B"], evaluate)
        assert result.contributions[0].name == "B"
        assert result.contributions[1].name == "A"


class TestAblationAttribution:
    """Tests for ablation-based signal attribution."""

    def test_basic_ablation(self):
        """Ablation correctly measures Sharpe drop per signal."""
        baseline = 2.0

        def ablation_fn(disabled: str) -> float:
            drops = {"A": 1.5, "B": 1.8, "C": 1.9}
            return drops[disabled]

        result = ablation_attribution(["A", "B", "C"], baseline, ablation_fn)
        assert result.method == "ablation"
        assert result.dominant == "A"  # A causes largest drop (0.5)

    def test_ablation_empty(self):
        """Empty signal list returns empty result."""
        result = ablation_attribution([], 1.0, lambda x: 0.0)
        assert result.contributions == ()

    def test_ablation_redundant(self):
        """Signal with <5% contribution is redundant."""
        baseline = 2.0

        def ablation_fn(disabled: str) -> float:
            return {"A": 0.5, "B": 1.95}.get(disabled, 2.0)

        result = ablation_attribution(["A", "B"], baseline, ablation_fn)
        assert "B" in result.redundant  # 0.05/2.0 = 2.5%

    def test_ablation_pct_of_total(self):
        """pct_of_total is correctly computed relative to baseline."""
        baseline = 2.0

        def ablation_fn(disabled: str) -> float:
            return 1.0  # each signal contributes 1.0

        result = ablation_attribution(["A", "B"], baseline, ablation_fn)
        for c in result.contributions:
            assert abs(c.pct_of_total - 50.0) < 1e-6


class TestAutoAttribution:
    """Tests for auto-selection between Shapley and Ablation."""

    def test_auto_selects_shapley_for_3_signals(self):
        """<=4 signals should use Shapley."""
        def evaluate(active: set[str]) -> float:
            return len(active) * 0.5

        result = auto_attribution(["A", "B", "C"], evaluate)
        assert result.method == "shapley"

    def test_auto_selects_ablation_for_5_signals(self):
        """5 signals should use Ablation."""
        names = ["A", "B", "C", "D", "E"]

        def evaluate(active: set[str]) -> float:
            return len(active) * 0.4

        def ablation_fn(disabled: str) -> float:
            return 1.6  # baseline 2.0 - 0.4

        result = auto_attribution(names, evaluate, ablation_fn=ablation_fn)
        assert result.method == "ablation"

    def test_auto_raises_without_ablation_fn(self):
        """Should raise ValueError if >4 signals and no ablation_fn."""
        names = ["A", "B", "C", "D", "E"]
        with pytest.raises(ValueError, match="ablation_fn is required"):
            auto_attribution(names, lambda x: 1.0)

    def test_auto_4_signals_uses_shapley(self):
        """Exactly 4 signals should use Shapley."""
        def evaluate(active: set[str]) -> float:
            return len(active) * 0.25

        result = auto_attribution(["A", "B", "C", "D"], evaluate)
        assert result.method == "shapley"


# ---------------------------------------------------------------------------
# Layer B: Horizon Attribution
# ---------------------------------------------------------------------------

class TestHorizonAttribution:
    """Tests for horizon (TSMOM factor) attribution."""

    def test_perfect_tsmom_fast(self):
        """Strategy that IS TSMOM 1M should have R²~1 and beta_fast~1."""
        np.random.seed(42)
        n = 500
        tsmom_1m = np.random.randn(n) * 0.01
        tsmom_3m = np.random.randn(n) * 0.01
        tsmom_12m = np.random.randn(n) * 0.01
        strategy = tsmom_1m  # perfect replication

        result = horizon_attribution(strategy, tsmom_1m, tsmom_3m, tsmom_12m)
        assert result.r_squared > 0.95
        assert abs(result.beta_fast - 1.0) < 0.15

    def test_mixed_horizons(self):
        """Strategy blending two horizons should show both betas."""
        np.random.seed(123)
        n = 500
        tsmom_1m = np.random.randn(n) * 0.01
        tsmom_3m = np.random.randn(n) * 0.01
        tsmom_12m = np.random.randn(n) * 0.01
        strategy = 0.5 * tsmom_1m + 0.5 * tsmom_3m

        result = horizon_attribution(strategy, tsmom_1m, tsmom_3m, tsmom_12m)
        assert result.r_squared > 0.9
        assert result.beta_fast > 0.3
        assert result.beta_medium > 0.3

    def test_independent_alpha(self):
        """Strategy with constant daily return has positive alpha."""
        n = 300
        tsmom_1m = np.random.randn(n) * 0.01
        tsmom_3m = np.random.randn(n) * 0.01
        tsmom_12m = np.random.randn(n) * 0.01
        strategy = np.full(n, 0.001)  # 0.1% daily alpha

        result = horizon_attribution(strategy, tsmom_1m, tsmom_3m, tsmom_12m)
        assert result.independent_alpha > 0.1  # annualized ~25%

    def test_fingerprint_sums_to_100(self):
        """Horizon fingerprint percentages should sum to ~100%."""
        np.random.seed(7)
        n = 300
        t1 = np.random.randn(n) * 0.01
        t3 = np.random.randn(n) * 0.01
        t12 = np.random.randn(n) * 0.01
        strat = 0.6 * t1 + 0.3 * t3 + 0.1 * t12

        result = horizon_attribution(strat, t1, t3, t12)
        total = sum(result.horizon_fingerprint.values())
        assert abs(total - 100.0) < 1e-6

    def test_empty_raises(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError):
            horizon_attribution(np.array([]), np.array([]), np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError):
            horizon_attribution(
                np.array([1, 2, 3]),
                np.array([1, 2]),
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
            )


# ---------------------------------------------------------------------------
# Layer C: Regime Attribution
# ---------------------------------------------------------------------------

class TestRegimeAttribution:
    """Tests for regime-based trade attribution."""

    def test_single_regime(self):
        """Single regime should be both best and worst."""
        pnls = np.array([10, -5, 20, -3])
        regimes = np.array(["trend", "trend", "trend", "trend"])

        result = regime_attribution(pnls, regimes)
        assert result.best_regime == "trend"
        assert result.worst_regime == "trend"
        assert result.regime_dependent is False

    def test_multiple_regimes(self):
        """Multiple regimes with different win rates."""
        pnls = np.array([10, 20, -5, -10, -15, -20])
        regimes = np.array(["trend", "trend", "mr", "mr", "crisis", "crisis"])

        result = regime_attribution(pnls, regimes)
        assert result.best_regime == "trend"  # 100% WR
        assert result.worst_regime in ("mr", "crisis")  # 0% WR

    def test_regime_dependent_flag(self):
        """Flag set when best/worst differ by >30pp."""
        pnls = np.array([10, 20, 30, -5, -10, -15])
        regimes = np.array(["trend", "trend", "trend", "crisis", "crisis", "crisis"])

        result = regime_attribution(pnls, regimes)
        assert result.regime_dependent is True  # 100% vs 0% = 100pp diff

    def test_regime_dependent_false_when_close(self):
        """Flag not set when win rates are close."""
        pnls = np.array([10, -5, 8, -3])
        regimes = np.array(["trend", "trend", "mr", "mr"])

        result = regime_attribution(pnls, regimes)
        assert result.regime_dependent is False  # both 50% WR

    def test_correct_stats(self):
        """Verify n_trades, win_rate, avg_pnl, total_pnl."""
        pnls = np.array([10, -5, 20])
        regimes = np.array(["A", "A", "A"])

        result = regime_attribution(pnls, regimes)
        stat = result.stats[0]
        assert stat.n_trades == 3
        assert abs(stat.win_rate - 2.0 / 3.0) < 1e-6
        assert abs(stat.avg_pnl - 25.0 / 3.0) < 1e-6
        assert abs(stat.total_pnl - 25.0) < 1e-6

    def test_empty_raises(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError):
            regime_attribution(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# Layer D: Baseline Decomposition
# ---------------------------------------------------------------------------

class TestBaselineDecomposition:
    """Tests for TSMOM + Carry baseline decomposition."""

    def test_pure_tsmom_strategy(self):
        """Strategy that IS TSMOM should have alpha ~0."""
        np.random.seed(42)
        n = 500
        tsmom = np.random.randn(n) * 0.01
        strategy = tsmom

        result = decompose_baseline(strategy, tsmom)
        assert result.r_squared > 0.99
        assert abs(result.independent_alpha) < 0.01

    def test_with_carry_component(self):
        """Strategy with carry component should show carry contribution."""
        np.random.seed(42)
        n = 500
        tsmom = np.random.randn(n) * 0.01
        carry = np.random.randn(n) * 0.005
        strategy = 0.5 * tsmom + 0.5 * carry

        result = decompose_baseline(strategy, tsmom, carry)
        assert result.r_squared > 0.9
        assert result.carry_pct > 0.0

    def test_no_carry(self):
        """Without carry, carry_beta_return and carry_pct should be 0."""
        np.random.seed(42)
        n = 300
        tsmom = np.random.randn(n) * 0.01
        strategy = tsmom + np.random.randn(n) * 0.001

        result = decompose_baseline(strategy, tsmom)
        assert result.carry_beta_return == 0.0
        assert result.carry_pct == 0.0

    def test_percentages_sum_near_100(self):
        """tsmom_pct + carry_pct + alpha_pct should sum to ~100%."""
        np.random.seed(42)
        n = 500
        tsmom = np.random.randn(n) * 0.01
        carry = np.random.randn(n) * 0.005
        strategy = 0.6 * tsmom + 0.3 * carry + np.full(n, 0.0005)

        result = decompose_baseline(strategy, tsmom, carry)
        total = result.tsmom_pct + result.carry_pct + result.alpha_pct
        assert abs(total - 100.0) < 1e-6

    def test_empty_raises(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError):
            decompose_baseline(np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError):
            decompose_baseline(np.array([1, 2, 3]), np.array([1, 2]))


# ---------------------------------------------------------------------------
# Layer E: Operational Attribution
# ---------------------------------------------------------------------------

class TestOperationalAttribution:
    """Tests for operational cost attribution."""

    def test_basic_decay(self):
        """Total decay = basic - industrial."""
        result = operational_attribution(2.0, 1.5)
        assert abs(result.total_decay - 0.5) < 1e-6

    def test_component_breakdown(self):
        """Individual components correctly computed."""
        components = {"slippage": 1.8, "spread": 1.7, "rollover": 1.95}
        result = operational_attribution(2.0, 1.2, components)
        assert abs(result.components["slippage"] - 0.2) < 1e-6
        assert abs(result.components["spread"] - 0.3) < 1e-6
        assert abs(result.components["rollover"] - 0.05) < 1e-6

    def test_no_components(self):
        """Without component_sharpes, components dict is empty."""
        result = operational_attribution(2.0, 1.5)
        assert result.components == {}

    def test_zero_decay(self):
        """No decay when basic == industrial."""
        result = operational_attribution(1.5, 1.5)
        assert abs(result.total_decay) < 1e-6


# ---------------------------------------------------------------------------
# Coverage Matrix
# ---------------------------------------------------------------------------

class TestCoverage:
    """Tests for regime coverage matrix."""

    def test_full_coverage(self):
        """All regimes covered -> coverage_score = 1.0."""
        data = {
            "strat_a": {"trend": 100, "mr": -50, "crisis": 10},
            "strat_b": {"trend": -20, "mr": 80, "crisis": -5},
        }
        result = regime_coverage(data)
        assert result.coverage_score == 1.0
        assert result.red_flags == ()

    def test_red_flag_detection(self):
        """Regime where all strategies lose -> RED FLAG."""
        data = {
            "strat_a": {"trend": 100, "crisis": -50},
            "strat_b": {"trend": 50, "crisis": -30},
        }
        result = regime_coverage(data)
        assert "crisis" in result.red_flags

    def test_partial_coverage(self):
        """coverage_score reflects fraction of covered regimes."""
        data = {
            "strat_a": {"trend": 100, "mr": -50, "crisis": -10},
        }
        result = regime_coverage(data)
        # trend covered, mr and crisis not -> 1/3
        assert abs(result.coverage_score - 1.0 / 3.0) < 1e-6

    def test_empty_strategies(self):
        """Empty input returns score 0."""
        result = regime_coverage({})
        assert result.coverage_score == 0.0
        assert result.red_flags == ()

    def test_zero_pnl_not_red_flag(self):
        """Zero PnL regime is not a red flag (not strictly negative)."""
        data = {
            "strat_a": {"trend": 0.0, "mr": -10},
        }
        result = regime_coverage(data)
        # trend has 0 PnL -> not positive (not covered), but not red flag
        assert "trend" not in result.red_flags
        # mr is negative for all strategies -> red flag
        assert "mr" in result.red_flags


# ---------------------------------------------------------------------------
# Alpha Decay
# ---------------------------------------------------------------------------

class TestAlphaDecay:
    """Tests for alpha decay detection."""

    def test_constant_ic_not_decaying(self):
        """Constant signal-return relationship should not decay."""
        np.random.seed(7)
        n = 500
        signals = np.random.randn(n)
        # Very strong positive relationship, no decay
        forward_returns = 2.0 * signals + np.random.randn(n) * 0.05

        result = detect_alpha_decay(signals, forward_returns, window=60, lookback=252)
        # IC should be consistently high with no significant negative trend
        assert result.is_decaying is False

    def test_declining_ic_is_decaying(self):
        """Signal that loses predictive power over time should decay."""
        np.random.seed(42)
        n = 600
        signals = np.random.randn(n)
        # Relationship degrades over time
        decay_factor = np.linspace(1.0, 0.0, n)
        forward_returns = decay_factor * signals + np.random.randn(n) * 0.3

        result = detect_alpha_decay(signals, forward_returns, window=60, lookback=400)
        assert result.ic_trend < 0.0

    def test_rolling_ic_length(self):
        """Rolling IC array should have correct length."""
        np.random.seed(42)
        n = 200
        signals = np.random.randn(n)
        forward_returns = np.random.randn(n)

        result = detect_alpha_decay(signals, forward_returns, window=60)
        assert len(result.rolling_ic) == n - 60 + 1

    def test_short_array_raises(self):
        """Arrays shorter than window should raise ValueError."""
        with pytest.raises(ValueError):
            detect_alpha_decay(np.array([1, 2, 3]), np.array([1, 2, 3]), window=60)

    def test_empty_raises(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError):
            detect_alpha_decay(np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        """Mismatched arrays should raise ValueError."""
        with pytest.raises(ValueError):
            detect_alpha_decay(np.array([1, 2, 3]), np.array([1, 2]))

    def test_half_life_present_when_decaying(self):
        """When decaying with positive current IC, half_life_bars should be set."""
        np.random.seed(99)
        n = 800
        signals = np.random.randn(n)
        decay_factor = np.linspace(1.0, 0.05, n)
        forward_returns = decay_factor * signals + np.random.randn(n) * 0.2

        result = detect_alpha_decay(signals, forward_returns, window=60, lookback=500)
        if result.is_decaying and result.rolling_ic[-1] > 0:
            assert result.half_life_bars is not None
            assert result.half_life_bars > 0


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

class TestReport:
    """Tests for attribution report generation."""

    def test_empty_report(self):
        """Report with no inputs still generates valid markdown."""
        report = generate_attribution_report()
        assert "# Strategy Attribution Report" in report

    def test_report_with_strategy_name(self):
        """Report includes strategy name in title."""
        report = generate_attribution_report(strategy_name="trend_fast_v1", symbol="RB")
        assert "trend_fast_v1 RB" in report

    def test_report_signal_section(self):
        """Report includes signal attribution section."""
        signal_result = shapley_attribution(
            ["A", "B"],
            lambda active: len(active) * 0.5,
        )
        report = generate_attribution_report(signal_result=signal_result)
        assert "## Signal Attribution" in report
        assert "shapley" in report

    def test_report_operational_section(self):
        """Report includes operational section."""
        op = operational_attribution(2.0, 1.5)
        report = generate_attribution_report(operational_result=op)
        assert "## Operational" in report
        assert "Industrial decay" in report

    def test_report_recommendations(self):
        """Report generates recommendations for issues."""
        op = operational_attribution(2.0, 0.5)  # 75% decay
        report = generate_attribution_report(operational_result=op)
        assert "Recommendations" in report
        assert "decay > 50%" in report
