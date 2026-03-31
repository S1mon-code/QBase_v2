"""Tests for portfolio construction module.

Covers: signal_blender, weights, constraints, selection, scorer,
        rebalance, retirement, regime_allocator.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from portfolio.signal_blender import (
    BlendedSignal,
    apply_direction_filter,
    apply_vol_targeting,
    blend_signals,
)
from portfolio.weights import (
    alpha_adjusted_weights,
    clip_and_redistribute,
    equal_weights,
    hrp_weights,
    inverse_volatility_weights,
)
from portfolio.constraints import check_horizon_balance
from portfolio.selection import select_strategies
from portfolio.scorer import PortfolioScore, score_portfolio
from portfolio.rebalance import RebalanceDecision, check_rebalance
from portfolio.retirement import RetirementCheck, check_retirement
from portfolio.regime_allocator import get_position_multiplier


# ============================================================
# signal_blender tests
# ============================================================


class TestBlendSignals:
    """Tests for blend_signals."""

    def test_basic_blend(self):
        """Two strategies with equal weights."""
        signals = {"a": 0.5, "b": -0.3}
        weights = {"a": 0.5, "b": 0.5}
        result = blend_signals(signals, weights)
        assert isinstance(result, BlendedSignal)
        assert result.net_signal == pytest.approx(0.1, abs=1e-9)
        assert set(result.active_strategies) == {"a", "b"}

    def test_empty_signals(self):
        """Empty signals returns zero."""
        result = blend_signals({}, {"a": 0.5})
        assert result.net_signal == 0.0
        assert result.active_strategies == ()

    def test_empty_weights(self):
        """Empty weights returns zero."""
        result = blend_signals({"a": 0.5}, {})
        assert result.net_signal == 0.0

    def test_no_overlap(self):
        """No overlap between signals and weights."""
        result = blend_signals({"a": 0.5}, {"b": 0.5})
        assert result.net_signal == 0.0

    def test_single_strategy(self):
        """Single strategy."""
        result = blend_signals({"a": 0.8}, {"a": 1.0})
        assert result.net_signal == pytest.approx(0.8)

    def test_clip_positive(self):
        """Net signal clipped to 1.0."""
        result = blend_signals({"a": 1.0, "b": 1.0}, {"a": 0.8, "b": 0.8})
        assert result.net_signal == 1.0

    def test_clip_negative(self):
        """Net signal clipped to -1.0."""
        result = blend_signals({"a": -1.0, "b": -1.0}, {"a": 0.8, "b": 0.8})
        assert result.net_signal == -1.0

    def test_strategy_weights_in_result(self):
        """Blended result contains used weights."""
        signals = {"a": 0.5, "b": 0.3}
        weights = {"a": 0.6, "b": 0.4, "c": 0.1}
        result = blend_signals(signals, weights)
        assert "a" in result.strategy_weights
        assert "b" in result.strategy_weights
        assert "c" not in result.strategy_weights

    def test_unnormalized_weights_signal_in_range(self):
        """Un-normalized weights (sum > 1) must not produce |signal| > 1 before clipping matters."""
        signals = {"a": 0.6, "b": 0.6}
        weights = {"a": 0.8, "b": 0.8}  # sum = 1.6
        result = blend_signals(signals, weights)
        assert -1.0 <= result.net_signal <= 1.0

    def test_unnormalized_weights_same_direction(self):
        """Un-normalized weights should produce the same signal direction as normalized weights."""
        signals = {"a": 0.5, "b": -0.2}
        # Normalized weights {a:0.5, b:0.5} → raw = 0.5*0.5 + 0.5*(-0.2) = 0.15
        # Un-normalized weights {a:2.0, b:2.0} sum=4.0 → normalized same as above
        result_norm = blend_signals(signals, {"a": 0.5, "b": 0.5})
        result_unnorm = blend_signals(signals, {"a": 2.0, "b": 2.0})
        assert result_norm.net_signal == pytest.approx(result_unnorm.net_signal)

    def test_normalized_weights_stored_in_result(self):
        """strategy_weights stored in BlendedSignal should be normalized (sum to 1)."""
        signals = {"a": 0.3, "b": 0.4}
        weights = {"a": 3.0, "b": 7.0}  # sum = 10.0
        result = blend_signals(signals, weights)
        stored_sum = sum(result.strategy_weights.values())
        assert stored_sum == pytest.approx(1.0)

    def test_proportional_weights_preserved(self):
        """Proportional relationships between weights should be preserved after normalization."""
        signals = {"a": 1.0, "b": 1.0}
        weights = {"a": 3.0, "b": 1.0}  # a gets 3x the weight of b
        result = blend_signals(signals, weights)
        assert result.strategy_weights["a"] == pytest.approx(0.75)
        assert result.strategy_weights["b"] == pytest.approx(0.25)


class TestDirectionFilter:
    """Tests for apply_direction_filter."""

    def test_long_positive_signal(self):
        assert apply_direction_filter(0.5, "long") == 0.5

    def test_long_negative_signal(self):
        assert apply_direction_filter(-0.5, "long") == 0.0

    def test_short_negative_signal(self):
        assert apply_direction_filter(-0.5, "short") == -0.5

    def test_short_positive_signal(self):
        assert apply_direction_filter(0.5, "short") == 0.0

    def test_neutral_passes_through(self):
        assert apply_direction_filter(0.5, "neutral") == 0.5
        assert apply_direction_filter(-0.5, "neutral") == -0.5

    def test_zero_signal_all_directions(self):
        assert apply_direction_filter(0.0, "long") == 0.0
        assert apply_direction_filter(0.0, "short") == 0.0
        assert apply_direction_filter(0.0, "neutral") == 0.0

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="Unknown direction"):
            apply_direction_filter(0.5, "invalid")


class TestVolTargeting:
    """Tests for apply_vol_targeting."""

    def test_scale_up(self):
        """Low realized vol scales up."""
        result = apply_vol_targeting(0.5, target_vol=0.20, realized_vol=0.10)
        assert result == pytest.approx(1.0)

    def test_scale_down(self):
        """High realized vol scales down."""
        result = apply_vol_targeting(0.5, target_vol=0.10, realized_vol=0.20)
        assert result == pytest.approx(0.25)

    def test_clip_low(self):
        """Scale factor clipped to minimum."""
        result = apply_vol_targeting(0.5, target_vol=0.01, realized_vol=0.50, clip_low=0.2)
        # scale = 0.02, clipped to 0.2, result = 0.5 * 0.2 = 0.1
        assert result == pytest.approx(0.1)

    def test_clip_high(self):
        """Scale factor clipped to maximum."""
        result = apply_vol_targeting(0.3, target_vol=1.0, realized_vol=0.01, clip_high=3.0)
        # scale = 100, clipped to 3.0, result = 0.3 * 3.0 = 0.9
        assert result == pytest.approx(0.9)

    def test_zero_realized_vol(self):
        """Zero realized vol returns signal unchanged."""
        result = apply_vol_targeting(0.5, target_vol=0.20, realized_vol=0.0)
        assert result == 0.5

    def test_output_clipped_to_one(self):
        """Output never exceeds 1.0."""
        result = apply_vol_targeting(0.8, target_vol=0.30, realized_vol=0.10)
        assert result <= 1.0


# ============================================================
# weights tests
# ============================================================


class TestEqualWeights:
    """Tests for equal_weights."""

    def test_three_strategies(self):
        w = equal_weights(["a", "b", "c"])
        assert len(w) == 3
        assert all(v == pytest.approx(1 / 3) for v in w.values())

    def test_single_strategy(self):
        w = equal_weights(["x"])
        assert w == {"x": 1.0}

    def test_empty(self):
        assert equal_weights([]) == {}


class TestInverseVolWeights:
    """Tests for inverse_volatility_weights."""

    def test_basic(self):
        w = inverse_volatility_weights({"a": 0.10, "b": 0.20})
        assert w["a"] > w["b"]
        assert sum(w.values()) == pytest.approx(1.0)

    def test_equal_vols(self):
        w = inverse_volatility_weights({"a": 0.15, "b": 0.15})
        assert w["a"] == pytest.approx(0.5)
        assert w["b"] == pytest.approx(0.5)

    def test_empty(self):
        assert inverse_volatility_weights({}) == {}

    def test_zero_vol_ignored(self):
        """Zero vol strategies get equal weight fallback."""
        w = inverse_volatility_weights({"a": 0.0, "b": 0.0})
        assert len(w) == 2


class TestHRPWeights:
    """Tests for hrp_weights."""

    def test_basic_hrp(self):
        """HRP with uncorrelated strategies gives roughly equal weights."""
        rng = np.random.RandomState(42)
        n_obs, n_strat = 200, 3
        returns = rng.randn(n_obs, n_strat) * 0.01
        names = ["s1", "s2", "s3"]
        w = hrp_weights(returns, names)
        assert len(w) == 3
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(v > 0 for v in w.values())

    def test_single_strategy(self):
        returns = np.array([[0.01], [0.02], [-0.01]])
        w = hrp_weights(returns, ["only"])
        assert w == {"only": 1.0}

    def test_empty(self):
        returns = np.empty((10, 0))
        assert hrp_weights(returns, []) == {}

    def test_correlated_strategies(self):
        """Highly correlated strategies should get lower combined weight than uncorrelated."""
        rng = np.random.RandomState(123)
        n = 200
        base = rng.randn(n) * 0.01
        # s1 and s2 are highly correlated; s3 is independent
        returns = np.column_stack([
            base + rng.randn(n) * 0.001,
            base + rng.randn(n) * 0.001,
            rng.randn(n) * 0.01,
        ])
        w = hrp_weights(returns, ["s1", "s2", "s3"])
        # s3 (independent) should get meaningful weight
        assert w["s3"] > 0.2

    def test_two_strategies(self):
        """Two strategies."""
        rng = np.random.RandomState(99)
        returns = rng.randn(100, 2) * 0.01
        w = hrp_weights(returns, ["a", "b"])
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


class TestAlphaAdjustedWeights:
    """Tests for alpha_adjusted_weights."""

    def test_basic_adjustment(self):
        hrp_w = {"a": 0.5, "b": 0.5}
        alphas = {"a": 0.05, "b": 0.02}
        win_rates = {"a": 0.7, "b": 0.6}
        w = alpha_adjusted_weights(hrp_w, alphas, win_rates)
        assert sum(w.values()) == pytest.approx(1.0)
        # "a" has higher alpha and win rate, should get more
        assert w["a"] > w["b"]

    def test_empty(self):
        assert alpha_adjusted_weights({}, {}, {}) == {}

    def test_no_overlap(self):
        """No common keys returns original weights."""
        w = alpha_adjusted_weights({"a": 1.0}, {"b": 0.05}, {"c": 0.7})
        assert w == {"a": 1.0}

    def test_zero_alpha(self):
        """All zero alphas fallback to equal weights."""
        hrp_w = {"a": 0.5, "b": 0.5}
        alphas = {"a": 0.0, "b": 0.0}
        win_rates = {"a": 0.6, "b": 0.6}
        w = alpha_adjusted_weights(hrp_w, alphas, win_rates)
        assert sum(w.values()) == pytest.approx(1.0)


class TestClipAndRedistribute:
    """Tests for clip_and_redistribute."""

    def test_no_clip_needed(self):
        w = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        result = clip_and_redistribute(w, max_weight=0.25)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_clip_single(self):
        w = {"a": 0.6, "b": 0.2, "c": 0.2}
        result = clip_and_redistribute(w, max_weight=0.4)
        assert result["a"] <= 0.4 + 1e-9
        assert sum(result.values()) == pytest.approx(1.0)

    def test_empty(self):
        assert clip_and_redistribute({}) == {}

    def test_all_above_max(self):
        """When all weights exceed max, they converge to equal."""
        w = {"a": 0.5, "b": 0.5}
        result = clip_and_redistribute(w, max_weight=0.25)
        assert sum(result.values()) == pytest.approx(1.0)


# ============================================================
# constraints tests
# ============================================================


class TestHorizonBalance:
    """Tests for check_horizon_balance."""

    def test_already_balanced(self):
        weights = {"a": 0.33, "b": 0.33, "c": 0.34}
        horizons = {"a": "fast", "b": "medium", "c": "slow"}
        result = check_horizon_balance(weights, horizons)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_under_weight_gets_boosted(self):
        """Slow horizon under-weight gets boosted from heavy horizon."""
        weights = {"a": 0.7, "b": 0.2, "c": 0.1}
        horizons = {"a": "fast", "b": "medium", "c": "slow"}
        result = check_horizon_balance(weights, horizons, min_per_horizon=0.15)
        slow_total = sum(result[s] for s, h in horizons.items() if h == "slow")
        assert slow_total >= 0.14  # approximately 0.15, may be slightly less due to normalization

    def test_all_same_horizon(self):
        """All same horizon: no rebalance possible (only 1 active horizon)."""
        weights = {"a": 0.5, "b": 0.5}
        horizons = {"a": "fast", "b": "fast"}
        result = check_horizon_balance(weights, horizons)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_empty(self):
        assert check_horizon_balance({}, {}) == {}

    def test_two_horizons(self):
        """Two horizons, one under-weight."""
        weights = {"a": 0.9, "b": 0.1}
        horizons = {"a": "fast", "b": "slow"}
        result = check_horizon_balance(weights, horizons, min_per_horizon=0.15)
        slow_w = result["b"]
        assert slow_w >= 0.14


# ============================================================
# selection tests
# ============================================================


class TestSelectStrategies:
    """Tests for select_strategies."""

    def _make_validation(
        self,
        regime_cv_verdict="PASS",
        industrial_sharpe=1.0,
        dsr=0.97,
        bootstrap_verdict="ROBUST",
    ):
        """Create a mock validation result."""
        return SimpleNamespace(
            regime_cv=SimpleNamespace(verdict=regime_cv_verdict),
            industrial=SimpleNamespace(industrial_sharpe=industrial_sharpe),
            deflated_sharpe=dsr,
            bootstrap=SimpleNamespace(verdict=bootstrap_verdict),
        )

    def test_all_pass(self):
        candidates = {
            "s1": {"validation": self._make_validation(), "alpha": 0.05, "activity": 0.01},
            "s2": {"validation": self._make_validation(), "alpha": 0.03, "activity": 0.005},
        }
        result = select_strategies(candidates)
        assert result == ["s1", "s2"]

    def test_regime_cv_fail(self):
        candidates = {
            "s1": {
                "validation": self._make_validation(regime_cv_verdict="FAIL"),
                "alpha": 0.05,
                "activity": 0.01,
            },
        }
        assert select_strategies(candidates) == []

    def test_low_industrial_sharpe(self):
        candidates = {
            "s1": {
                "validation": self._make_validation(industrial_sharpe=-0.1),
                "alpha": 0.05,
                "activity": 0.01,
            },
        }
        assert select_strategies(candidates) == []

    def test_low_dsr(self):
        candidates = {
            "s1": {
                "validation": self._make_validation(dsr=0.90),
                "alpha": 0.05,
                "activity": 0.01,
            },
        }
        assert select_strategies(candidates) == []

    def test_fragile_bootstrap(self):
        candidates = {
            "s1": {
                "validation": self._make_validation(bootstrap_verdict="FRAGILE"),
                "alpha": 0.05,
                "activity": 0.01,
            },
        }
        assert select_strategies(candidates) == []

    def test_low_activity(self):
        candidates = {
            "s1": {
                "validation": self._make_validation(),
                "alpha": 0.05,
                "activity": 0.0001,
            },
        }
        assert select_strategies(candidates) == []

    def test_negative_alpha(self):
        candidates = {
            "s1": {
                "validation": self._make_validation(),
                "alpha": -0.01,
                "activity": 0.01,
            },
        }
        assert select_strategies(candidates) == []

    def test_no_validation(self):
        candidates = {"s1": {"validation": None, "alpha": 0.05, "activity": 0.01}}
        assert select_strategies(candidates) == []


# ============================================================
# scorer tests
# ============================================================


class TestScorer:
    """Tests for score_portfolio."""

    def _good_params(self) -> dict:
        return dict(
            sharpe=1.5, calmar=2.0, max_dd=-0.10, dd_duration=30, cvar_95=-0.02,
            avg_indep_alpha=0.05, horizon_diversity=0.8, vs_tsmom_excess=0.05,
            avg_correlation=0.15, dd_overlap=0.15, portfolio_vs_best=1.2, pos_sharpe_pct=0.9,
            bootstrap_ci_width=0.3, core_pct=0.8, permutation_p=0.01,
            n_strategies=8, max_weight=0.15, industrial_decay=0.1,
        )

    def _bad_params(self) -> dict:
        return dict(
            sharpe=0.0, calmar=0.0, max_dd=-0.50, dd_duration=250, cvar_95=-0.15,
            avg_indep_alpha=0.0, horizon_diversity=0.0, vs_tsmom_excess=0.0,
            avg_correlation=0.8, dd_overlap=0.8, portfolio_vs_best=0.5, pos_sharpe_pct=0.0,
            bootstrap_ci_width=2.0, core_pct=0.0, permutation_p=0.2,
            n_strategies=1, max_weight=1.0, industrial_decay=0.5,
        )

    def test_good_portfolio_grade(self):
        result = score_portfolio(**self._good_params())
        assert isinstance(result, PortfolioScore)
        assert result.grade in ("A+", "A", "A-")
        assert result.passed is True

    def test_bad_portfolio_grade(self):
        result = score_portfolio(**self._bad_params())
        assert result.grade in ("D/F", "C")
        assert result.passed is False

    def test_boundary_75(self):
        """Score near 75 boundary."""
        result = score_portfolio(**self._good_params())
        assert result.total >= 75.0

    def test_boundary_90(self):
        """Perfect scores yield A+."""
        params = self._good_params()
        params["sharpe"] = 2.5
        params["calmar"] = 3.5
        params["max_dd"] = -0.05
        params["dd_duration"] = 10
        params["cvar_95"] = -0.01
        params["avg_indep_alpha"] = 0.1
        params["horizon_diversity"] = 1.0
        params["vs_tsmom_excess"] = 0.1
        params["avg_correlation"] = 0.0
        params["dd_overlap"] = 0.0
        params["portfolio_vs_best"] = 1.5
        params["pos_sharpe_pct"] = 1.0
        params["bootstrap_ci_width"] = 0.0
        params["core_pct"] = 1.0
        params["permutation_p"] = 0.0
        params["n_strategies"] = 12
        params["max_weight"] = 0.1
        params["industrial_decay"] = 0.0
        result = score_portfolio(**params)
        assert result.grade == "A+"

    def test_dimensions_present(self):
        result = score_portfolio(**self._good_params())
        expected_dims = {"return_risk", "signal_quality", "efficiency", "robustness", "operability"}
        assert set(result.dimensions.keys()) == expected_dims

    def test_metrics_stored(self):
        result = score_portfolio(**self._good_params())
        assert "sharpe" in result.metrics
        assert "calmar" in result.metrics

    def test_total_in_range(self):
        result = score_portfolio(**self._good_params())
        assert 0 <= result.total <= 100


# ============================================================
# rebalance tests
# ============================================================


class TestRebalance:
    """Tests for check_rebalance."""

    def test_monthly_due(self):
        result = check_rebalance(date(2024, 1, 1), date(2024, 2, 1))
        assert isinstance(result, RebalanceDecision)
        assert result.should_rebalance is True
        assert result.days_since_last == 31

    def test_monthly_not_due(self):
        result = check_rebalance(date(2024, 1, 1), date(2024, 1, 15))
        assert result.should_rebalance is False
        assert result.days_since_last == 14

    def test_weekly_due(self):
        result = check_rebalance(date(2024, 1, 1), date(2024, 1, 8), frequency="weekly")
        assert result.should_rebalance is True

    def test_weekly_not_due(self):
        result = check_rebalance(date(2024, 1, 1), date(2024, 1, 5), frequency="weekly")
        assert result.should_rebalance is False

    def test_strategy_change_triggers(self):
        result = check_rebalance(
            date(2024, 1, 1), date(2024, 1, 2), strategy_changed=True
        )
        assert result.should_rebalance is True
        assert "changed" in result.reason.lower()

    def test_invalid_frequency(self):
        with pytest.raises(ValueError, match="Unknown frequency"):
            check_rebalance(date(2024, 1, 1), date(2024, 2, 1), frequency="daily")


# ============================================================
# retirement tests
# ============================================================


class TestRetirement:
    """Tests for check_retirement."""

    def test_normal(self):
        result = check_retirement("s1", 0.5, 0, 0.8, -0.05, -0.10)
        assert isinstance(result, RetirementCheck)
        assert result.action == "normal"

    def test_observe_6m_sharpe(self):
        """6m Sharpe < 0 triggers observe."""
        result = check_retirement("s1", -0.1, 0, 0.5, -0.05, -0.10)
        assert result.action == "observe"

    def test_observe_consecutive_loss(self):
        """3+ consecutive loss months triggers observe."""
        result = check_retirement("s1", 0.1, 3, 0.5, -0.05, -0.10)
        assert result.action == "observe"

    def test_remove_12m_sharpe(self):
        """12m Sharpe < -0.5 triggers remove."""
        result = check_retirement("s1", -0.3, 1, -0.6, -0.05, -0.10)
        assert result.action == "remove"

    def test_immediate_remove_dd(self):
        """DD > 1.5x backtest max DD triggers immediate remove."""
        result = check_retirement("s1", 0.5, 0, 0.8, -0.20, -0.10)
        # threshold = -0.10 * 1.5 = -0.15, current = -0.20 < -0.15
        assert result.action == "immediate_remove"

    def test_immediate_remove_takes_priority(self):
        """Immediate remove overrides other conditions."""
        result = check_retirement("s1", -0.3, 5, -0.8, -0.20, -0.10)
        assert result.action == "immediate_remove"

    def test_remove_priority_over_observe(self):
        """Remove overrides observe."""
        result = check_retirement("s1", -0.3, 4, -0.6, -0.05, -0.10)
        assert result.action == "remove"

    def test_boundary_dd_exact_threshold(self):
        """DD exactly at 1.5x threshold is not immediate remove."""
        result = check_retirement("s1", 0.5, 0, 0.8, -0.15, -0.10)
        # -0.15 is NOT < -0.15, so not triggered
        assert result.action == "normal"

    def test_strategy_name_preserved(self):
        result = check_retirement("my_strategy", 0.5, 0, 0.8, -0.05, -0.10)
        assert result.strategy == "my_strategy"


# ============================================================
# regime_allocator tests
# ============================================================


class TestRegimeAllocator:
    """Tests for get_position_multiplier."""

    def test_crisis_multiplier(self):
        assert get_position_multiplier("crisis") == 0.5

    def test_strong_trend_multiplier(self):
        assert get_position_multiplier("strong_trend") == 1.0

    def test_mild_trend_multiplier(self):
        assert get_position_multiplier("mild_trend") == 1.0

    def test_mean_reversion_multiplier(self):
        assert get_position_multiplier("mean_reversion") == 1.0

    def test_unknown_regime(self):
        """Unknown regimes default to 1.0."""
        assert get_position_multiplier("unknown") == 1.0
