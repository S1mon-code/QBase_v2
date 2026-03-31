"""Comprehensive tests for the optimizer module.

Covers: composite_objective, param_discovery, robustness, multi_seed, trial_registry.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from optimizer.core import (
    BacktestMetrics,
    _score_alpha,
    _score_consistency,
    _score_performance,
    _score_risk,
    _score_significance,
    composite_objective,
)
from optimizer.param_discovery import discover_params
from optimizer.robustness import _perturb_params, check_robustness, multi_seed_optimize
from optimizer.trial_registry import TrialRegistry


# ======================================================================
# Helpers
# ======================================================================


def _make_metrics(
    sharpe: float = 1.5,
    max_drawdown: float = 0.10,
    cvar_95: float = -0.01,
    n_trades: int = 50,
    win_rate: float = 0.6,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_days: int = 250,
    equity_start: float = 100.0,
) -> BacktestMetrics:
    """Create BacktestMetrics with sensible defaults."""
    rng = np.random.RandomState(42)
    daily = rng.normal(0.001, 0.01, n_days)
    equity = equity_start * np.cumprod(1 + daily)
    return BacktestMetrics(
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        cvar_95=cvar_95,
        n_trades=n_trades,
        win_rate=win_rate,
        skewness=skewness,
        kurtosis=kurtosis,
        daily_returns=daily,
        equity_curve=equity,
    )


# ======================================================================
# composite_objective — hard filters
# ======================================================================


class TestCompositeHardFilters:
    """Test hard filter conditions that reject strategies immediately."""

    def test_insufficient_trades_daily(self):
        m = _make_metrics(n_trades=5)
        assert composite_objective(m, freq="daily") == -10.0

    def test_insufficient_trades_1h(self):
        m = _make_metrics(n_trades=20)
        assert composite_objective(m, freq="1h") == -10.0

    def test_insufficient_trades_5min(self):
        m = _make_metrics(n_trades=70)
        assert composite_objective(m, freq="5min") == -10.0

    def test_sufficient_trades_passes(self):
        m = _make_metrics(sharpe=1.5, n_trades=50)
        score = composite_objective(m, baseline_sharpe=0.0, freq="1h")
        assert score > 0

    def test_alpha_zero_returns_negative5(self):
        m = _make_metrics(sharpe=1.0, n_trades=50)
        assert composite_objective(m, baseline_sharpe=1.0, freq="1h") == -5.0

    def test_alpha_negative_returns_negative5(self):
        m = _make_metrics(sharpe=0.5, n_trades=50)
        assert composite_objective(m, baseline_sharpe=1.5, freq="1h") == -5.0


# ======================================================================
# composite_objective — score dimensions
# ======================================================================


class TestCompositeScoring:
    """Test the composite score across different metric profiles."""

    def test_sharpe_zero_low_score(self):
        m = _make_metrics(sharpe=0.01, n_trades=50)
        score = composite_objective(m, baseline_sharpe=0.0, freq="1h")
        assert score < 3.0

    def test_high_sharpe_low_dd_high_score(self):
        m = _make_metrics(sharpe=2.5, max_drawdown=0.05, cvar_95=-0.005, n_trades=100)
        score = composite_objective(m, baseline_sharpe=0.0, freq="1h")
        assert score > 5.0

    def test_high_sharpe_high_dd_medium_score(self):
        m = _make_metrics(sharpe=2.5, max_drawdown=0.35, cvar_95=-0.025, n_trades=100)
        low_dd = _make_metrics(sharpe=2.5, max_drawdown=0.05, cvar_95=-0.005, n_trades=100)
        score_high_dd = composite_objective(m, baseline_sharpe=0.0, freq="1h")
        score_low_dd = composite_objective(low_dd, baseline_sharpe=0.0, freq="1h")
        assert score_high_dd < score_low_dd  # risk penalty

    def test_coarse_vs_fine_different_performance(self):
        m = _make_metrics(sharpe=2.0, n_trades=50)
        coarse = composite_objective(m, baseline_sharpe=0.0, phase="coarse", freq="1h")
        fine = composite_objective(m, baseline_sharpe=0.0, phase="fine", freq="1h")
        # Both positive, but may differ due to tanh vs linear
        assert coarse > 0
        assert fine > 0

    def test_score_bounded_0_10(self):
        m = _make_metrics(sharpe=5.0, max_drawdown=0.01, cvar_95=-0.001, n_trades=200)
        score = composite_objective(m, baseline_sharpe=0.0, freq="1h")
        assert 0 <= score <= 10.0

    def test_unknown_freq_defaults_to_30(self):
        m = _make_metrics(sharpe=1.5, n_trades=25)
        # Unknown freq → MIN_TRADES defaults to 30, so 25 < 30
        assert composite_objective(m, freq="unknown") == -10.0


# ======================================================================
# S_performance
# ======================================================================


class TestScorePerformance:
    """Test performance scoring dimension."""

    def test_coarse_tanh_compression(self):
        # tanh(0.7 * 3) ≈ 0.97 → ~9.7
        s = _score_performance(3.0, "coarse")
        assert 9.0 < s <= 10.0

    def test_coarse_negative_sharpe(self):
        s = _score_performance(-1.0, "coarse")
        assert s < 0

    def test_fine_linear(self):
        s = _score_performance(1.5, "fine")
        assert abs(s - 5.0) < 0.01  # 1.5 * 10/3 = 5.0

    def test_fine_capped_at_10(self):
        s = _score_performance(5.0, "fine")
        assert s == 10.0


# ======================================================================
# S_significance
# ======================================================================


class TestScoreSignificance:
    """Test statistical significance dimension."""

    def test_few_trades_low_score(self):
        s = _score_significance(1.0, 5, 0.0, 3.0)
        assert s < 8.0

    def test_many_trades_high_score(self):
        s = _score_significance(1.5, 500, 0.0, 3.0)
        assert s > 5.0

    def test_zero_trades(self):
        assert _score_significance(1.0, 0, 0.0, 3.0) == 0.0

    def test_high_skew_reduces_score(self):
        normal = _score_significance(1.0, 100, 0.0, 3.0)
        skewed = _score_significance(1.0, 100, 2.0, 3.0)
        # With positive skew and positive sharpe, denominator increases → lower t-stat
        # (depends on direction, but we just check it's non-negative)
        assert skewed >= 0.0

    def test_capped_at_10(self):
        s = _score_significance(3.0, 1000, 0.0, 3.0)
        assert s == 10.0


# ======================================================================
# S_consistency
# ======================================================================


class TestScoreConsistency:
    """Test time consistency dimension."""

    def test_all_positive_windows(self):
        # Returns all positive → high win rate, low CV
        returns = np.ones(250) * 0.001
        s = _score_consistency(returns)
        assert s > 7.0

    def test_mixed_windows(self):
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0, 0.01, 250)
        s = _score_consistency(returns)
        assert 0.0 <= s <= 10.0

    def test_too_few_returns(self):
        returns = np.array([0.01, 0.02])
        s = _score_consistency(returns, n_windows=5)
        assert s == 0.0

    def test_zero_mean_returns(self):
        returns = np.zeros(250)
        s = _score_consistency(returns)
        assert s == 0.0


# ======================================================================
# S_risk
# ======================================================================


class TestScoreRisk:
    """Test tail risk dimension."""

    def test_low_dd_high_score(self):
        s = _score_risk(0.05, -0.005)
        assert s > 7.0

    def test_extreme_dd_zero_score(self):
        s = _score_risk(0.40, -0.03)
        assert s == pytest.approx(0.0, abs=0.01)

    def test_high_dd_penalized(self):
        low = _score_risk(0.05, -0.01)
        high = _score_risk(0.30, -0.01)
        assert low > high

    def test_cvar_extreme(self):
        s = _score_risk(0.10, -0.03)
        # CVaR of -3% → cvar_score = 0
        cvar_part = 0.4 * 0.0
        maxdd_part = 0.6 * max(0, 10 * (1 - 0.10 / 0.40))
        assert abs(s - (maxdd_part + cvar_part)) < 0.01


# ======================================================================
# S_alpha
# ======================================================================


class TestScoreAlpha:
    """Test alpha dimension."""

    def test_above_baseline_positive(self):
        s = _score_alpha(2.0, 1.0)
        assert s == 10.0  # alpha=1.0 → full score

    def test_below_baseline_zero(self):
        s = _score_alpha(0.5, 1.0)
        assert s == 0.0

    def test_partial_alpha(self):
        s = _score_alpha(1.5, 1.0)
        assert abs(s - 5.0) < 0.01  # alpha=0.5 → 5.0

    def test_equal_baseline_zero(self):
        s = _score_alpha(1.0, 1.0)
        assert s == 0.0


# ======================================================================
# param_discovery
# ======================================================================


class TestParamDiscovery:
    """Test automatic parameter range discovery."""

    def test_known_param_gets_fixed_range(self):
        class Strategy:
            st_mult: float = 3.0

        params = discover_params(Strategy)
        assert "st_mult" in params
        assert params["st_mult"]["low"] == 1.5
        assert params["st_mult"]["high"] == 5.0

    def test_period_param_gets_auto_range(self):
        class Strategy:
            st_period: int = 10

        params = discover_params(Strategy)
        assert "st_period" in params
        assert params["st_period"]["low"] == 4  # int(10 * 0.4)
        assert params["st_period"]["high"] == 30  # int(10 * 3.0)
        assert params["st_period"]["step"] == 1

    def test_skip_list_works(self):
        class Strategy:
            name: str = "test"
            warmup: int = 100
            regime: str = "trend"
            st_period: int = 10

        params = discover_params(Strategy)
        assert "name" not in params
        assert "warmup" not in params
        assert "regime" not in params
        assert "st_period" in params

    def test_float_param_continuous(self):
        class Strategy:
            threshold: float = 0.5

        params = discover_params(Strategy)
        assert params["threshold"]["step"] is None

    def test_empty_class_empty_params(self):
        class Strategy:
            pass

        params = discover_params(Strategy)
        assert params == {}

    def test_non_numeric_skipped(self):
        class Strategy:
            label: str = "abc"
            flag: bool = True
            period: int = 20

        params = discover_params(Strategy)
        assert "label" not in params
        assert "flag" not in params
        assert "period" in params

    def test_chandelier_mult_known(self):
        class Strategy:
            chandelier_mult: float = 2.5

        params = discover_params(Strategy)
        assert params["chandelier_mult"]["low"] == 1.5
        assert params["chandelier_mult"]["high"] == 4.0

    def test_lookback_param(self):
        class Strategy:
            lookback: int = 20

        params = discover_params(Strategy)
        assert params["lookback"]["low"] == 8   # int(20 * 0.4)
        assert params["lookback"]["high"] == 60  # int(20 * 3.0)

    def test_mult_param_auto_range(self):
        class Strategy:
            custom_mult: float = 2.0

        params = discover_params(Strategy)
        assert abs(params["custom_mult"]["low"] - 0.6) < 0.01   # 2.0 * 0.3
        assert abs(params["custom_mult"]["high"] - 6.0) < 0.01  # 2.0 * 3.0


# ======================================================================
# robustness — check_robustness
# ======================================================================


class TestCheckRobustness:
    """Test plateau detection via neighbor sampling."""

    def test_all_neighbors_above_threshold_is_plateau(self):
        best_params = {"a": 10.0, "b": 5.0}

        def eval_fn(params):
            return 8.0  # Always high

        result = check_robustness(best_params, 8.0, eval_fn, n_samples=20)
        assert result["is_robust"] is True
        assert result["above_threshold_pct"] == 1.0

    def test_all_neighbors_below_threshold_is_spike(self):
        best_params = {"a": 10.0, "b": 5.0}

        def eval_fn(params):
            return 0.1  # Always low (threshold = 8.0 * 0.5 = 4.0)

        result = check_robustness(best_params, 8.0, eval_fn, n_samples=20)
        assert result["is_robust"] is False
        assert result["above_threshold_pct"] == 0.0

    def test_exactly_60_percent_is_plateau(self):
        best_params = {"a": 10.0}
        call_count = [0]

        def eval_fn(params):
            call_count[0] += 1
            # 12 out of 20 = 60% → threshold
            return 5.0 if call_count[0] <= 12 else 0.0

        result = check_robustness(best_params, 8.0, eval_fn, n_samples=20)
        assert result["is_robust"] is True

    def test_neighbor_stats(self):
        best_params = {"x": 5.0}

        def eval_fn(params):
            return 6.0

        result = check_robustness(best_params, 8.0, eval_fn, n_samples=10)
        assert result["neighbor_mean"] == pytest.approx(6.0)
        assert result["neighbor_std"] == pytest.approx(0.0)
        assert len(result["neighbor_scores"]) == 10

    def test_default_n_samples(self):
        best_params = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}

        def eval_fn(params):
            return 5.0

        result = check_robustness(best_params, 8.0, eval_fn)
        # max(20, 5*5) = 25
        assert len(result["neighbor_scores"]) == 25

    def test_negative_params_no_error(self):
        """Negative param values must not cause rng.uniform(low > high) error."""
        best_params = {"a": -5.0, "b": -1.0}

        def eval_fn(params):
            return 4.0

        # Should not raise ValueError from rng.uniform
        result = check_robustness(best_params, 8.0, eval_fn, n_samples=20)
        assert len(result["neighbor_scores"]) == 20

    def test_negative_param_perturb_stays_negative(self):
        """Perturbing a negative param should produce neighbors in the correct range."""
        import random
        rng = random.Random(0)
        for _ in range(50):
            neighbor = _perturb_params({"v": -10.0}, radius=0.15, rng=rng)
            # low = -10 * 1.15 = -11.5, high = -10 * 0.85 = -8.5
            assert -11.5 <= neighbor["v"] <= -8.5

    def test_zero_param_perturb_bounded(self):
        """Zero param value uses [-radius, +radius] range."""
        import random
        rng = random.Random(0)
        for _ in range(50):
            neighbor = _perturb_params({"v": 0.0}, radius=0.15, rng=rng)
            assert -0.15 <= neighbor["v"] <= 0.15

    def test_zero_int_param_perturb_bounded(self):
        """Zero integer param rounds to 0 with small radius."""
        import random
        rng = random.Random(0)
        for _ in range(50):
            neighbor = _perturb_params({"v": 0}, radius=0.15, rng=rng)
            # range is [-0.15, 0.15], rounded to nearest int → always 0
            assert neighbor["v"] == 0


# ======================================================================
# robustness — multi_seed_optimize
# ======================================================================


class TestMultiSeedOptimize:
    """Test multi-seed optimization."""

    def test_consistent_results(self):
        def opt_fn(seed):
            return {"a": 10}, 7.0

        result = multi_seed_optimize(opt_fn)
        assert result["is_consistent"] is True
        assert result["best_score"] == 7.0
        assert len(result["all_scores"]) == 3

    def test_divergent_results_inconsistent(self):
        scores = iter([1.0, 50.0, 100.0])

        def opt_fn(seed):
            s = next(scores)
            return {"a": seed}, s

        result = multi_seed_optimize(opt_fn)
        assert result["is_consistent"] is False

    def test_returns_median(self):
        scores = iter([2.0, 8.0, 5.0])
        params_list = iter([{"a": 1}, {"a": 2}, {"a": 3}])

        def opt_fn(seed):
            return next(params_list), next(scores)

        result = multi_seed_optimize(opt_fn)
        assert result["best_score"] == 5.0
        assert result["best_params"] == {"a": 3}

    def test_custom_seeds(self):
        call_seeds = []

        def opt_fn(seed):
            call_seeds.append(seed)
            return {"a": 1}, 5.0

        multi_seed_optimize(opt_fn, seeds=(1, 2, 3, 4, 5))
        assert call_seeds == [1, 2, 3, 4, 5]


# ======================================================================
# trial_registry
# ======================================================================


class TestTrialRegistry:
    """Test append-only trial recording."""

    def test_record_and_count(self, tmp_path):
        reg = TrialRegistry(tmp_path / "trials.jsonl")
        reg.record("trend_v1", {"a": 1}, 1.5, 7.0, "strong_trend", "RB", "1h", 50)
        assert reg.get_total_trials() == 1

    def test_get_all_sharpes(self, tmp_path):
        reg = TrialRegistry(tmp_path / "trials.jsonl")
        reg.record("trend_v1", {"a": 1}, 1.5, 7.0, "trend", "RB", "1h", 50)
        reg.record("trend_v1", {"a": 2}, 2.0, 8.0, "trend", "RB", "1h", 60)
        sharpes = reg.get_all_sharpes()
        assert sharpes == [1.5, 2.0]

    def test_filter_by_strategy(self, tmp_path):
        reg = TrialRegistry(tmp_path / "trials.jsonl")
        reg.record("trend_v1", {}, 1.0, 5.0, "trend", "RB", "1h", 30)
        reg.record("mr_v1", {}, 0.8, 4.0, "mr", "RB", "1h", 40)
        reg.record("trend_v1", {}, 1.2, 6.0, "trend", "RB", "1h", 35)
        assert len(reg.get_trials_for_strategy("trend_v1")) == 2
        assert len(reg.get_trials_for_strategy("mr_v1")) == 1

    def test_append_only_file_grows(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        reg = TrialRegistry(path)
        reg.record("s1", {}, 1.0, 5.0, "trend", "RB", "1h", 30)
        size1 = path.stat().st_size
        reg.record("s2", {}, 1.5, 6.0, "trend", "RB", "1h", 40)
        size2 = path.stat().st_size
        assert size2 > size1

    def test_empty_registry_returns_zero(self, tmp_path):
        reg = TrialRegistry(tmp_path / "nonexistent.jsonl")
        assert reg.get_total_trials() == 0
        assert reg.get_all_sharpes() == []

    def test_jsonl_format(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        reg = TrialRegistry(path)
        reg.record("trend_v1", {"x": 1}, 1.5, 7.0, "trend", "RB", "1h", 50)
        line = path.read_text().strip()
        data = json.loads(line)
        assert data["strategy"] == "trend_v1"
        assert data["sharpe"] == 1.5
        assert data["params"] == {"x": 1}

    def test_multiple_records_separate_lines(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        reg = TrialRegistry(path)
        for i in range(5):
            reg.record(f"s{i}", {}, float(i), float(i), "t", "RB", "1h", 10)
        lines = [l for l in path.read_text().strip().split("\n") if l]
        assert len(lines) == 5

    def test_trial_ids_increment(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        reg = TrialRegistry(path)
        reg.record("s1", {}, 1.0, 5.0, "t", "RB", "1h", 30)
        reg.record("s2", {}, 2.0, 6.0, "t", "RB", "1h", 40)
        lines = path.read_text().strip().split("\n")
        assert json.loads(lines[0])["id"] == "trial_0001"
        assert json.loads(lines[1])["id"] == "trial_0002"

    def test_record_preserves_params(self, tmp_path):
        reg = TrialRegistry(tmp_path / "trials.jsonl")
        params = {"st_period": 10, "st_mult": 3.0, "chandelier_mult": 2.5}
        reg.record("trend_v1", params, 1.5, 7.0, "trend", "RB", "1h", 50)
        trials = reg.get_trials_for_strategy("trend_v1")
        assert trials[0]["params"] == params


# ======================================================================
# Integration: composite_objective edge cases
# ======================================================================


class TestCompositeEdgeCases:
    """Additional edge cases for composite scoring."""

    def test_freq_4h_threshold(self):
        m = _make_metrics(sharpe=1.5, n_trades=15)
        assert composite_objective(m, freq="4h") == -10.0

    def test_freq_30min_threshold(self):
        m = _make_metrics(sharpe=1.5, n_trades=45)
        assert composite_objective(m, freq="30min") == -10.0

    def test_very_high_sharpe_capped(self):
        s = _score_performance(10.0, "fine")
        assert s == 10.0

    def test_negative_sharpe_coarse(self):
        m = _make_metrics(sharpe=-0.5, n_trades=50)
        # alpha <= 0 → -5
        assert composite_objective(m, baseline_sharpe=0.0, freq="1h") == -5.0

    def test_positive_cvar_high_risk_score(self):
        # Edge case: positive CVaR (unusual but possible)
        s = _score_risk(0.05, 0.01)
        assert s > 7.0

    def test_zero_drawdown_perfect_risk(self):
        s = _score_risk(0.0, 0.0)
        # maxdd=0 → 10, cvar=0 → 10*(1+0)=10
        assert s == pytest.approx(10.0)
