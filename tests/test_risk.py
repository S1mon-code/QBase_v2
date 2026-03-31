"""Comprehensive tests for the risk management module.

Covers: ChandelierExit, VolTargeting, PositionSizer, DirectionalFilter,
        VolClassifier, PortfolioStops.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from risk.chandelier import ChandelierExit
from risk.vol_targeting import (
    realized_vol,
    vol_scale,
    atr_percentile,
    extreme_vol_adjustment,
    VolTargeting,
)
from risk.position_sizer import calc_lots, PositionSizer
from risk.directional_filter import load_direction, filter_signal, DirectionalFilter
from risk.vol_classifier import classify_vol
from risk.portfolio_stops import PortfolioStops


# ======================================================================
# ChandelierExit
# ======================================================================


class TestChandelierExitLong:
    """Long-side Chandelier Exit tests."""

    def test_stop_rises_on_new_highs(self):
        """Long stop must rise (or stay) when price makes new highs."""
        ce = ChandelierExit(atr_mult=2.0, regime="strong_trend")
        stops = []
        for i in range(10):
            h = 100.0 + i * 2
            l = 98.0 + i * 2
            c = 99.0 + i * 2
            ce.update(h, l, c, atr=5.0, side=1)
            stops.append(ce.get_stop())
        for i in range(1, len(stops)):
            assert stops[i] >= stops[i - 1], f"Stop fell at bar {i}"

    def test_stop_never_falls_on_pullback(self):
        """Long stop must not decrease when price pulls back."""
        ce = ChandelierExit(atr_mult=2.0, regime="trending")
        # Rising phase
        for i in range(5):
            ce.update(110.0 + i, 108.0 + i, 109.0 + i, atr=3.0, side=1)
        peak_stop = ce.get_stop()
        # Pullback phase
        for i in range(5):
            ce.update(112.0 - i, 110.0 - i, 111.0 - i, atr=3.0, side=1)
            assert ce.get_stop() >= peak_stop

    def test_initial_stop_correct(self):
        """First bar stop = high - mult*atr for long."""
        ce = ChandelierExit(atr_mult=3.0, regime="strong_trend")
        ce.update(high=105.0, low=100.0, close=103.0, atr=2.0, side=1)
        assert ce.get_stop() == pytest.approx(105.0 - 3.0 * 2.0)

    def test_is_stopped_long(self):
        """Long position should be stopped when close <= stop."""
        ce = ChandelierExit(atr_mult=2.0, regime="trending")
        ce.update(100.0, 98.0, 99.0, atr=5.0, side=1)
        stop = ce.get_stop()
        assert not ce.is_stopped(stop + 0.01, 1)
        assert ce.is_stopped(stop, 1)
        assert ce.is_stopped(stop - 1.0, 1)


class TestChandelierExitShort:
    """Short-side Chandelier Exit tests."""

    def test_stop_falls_on_new_lows(self):
        """Short stop must fall (or stay) when price makes new lows."""
        ce = ChandelierExit(atr_mult=2.0, regime="strong_trend")
        stops = []
        for i in range(10):
            h = 102.0 - i * 2
            l = 100.0 - i * 2
            c = 101.0 - i * 2
            ce.update(h, l, c, atr=5.0, side=-1)
            stops.append(ce.get_stop())
        for i in range(1, len(stops)):
            assert stops[i] <= stops[i - 1], f"Stop rose at bar {i}"

    def test_stop_never_rises_on_bounce(self):
        """Short stop must not increase when price bounces."""
        ce = ChandelierExit(atr_mult=2.0, regime="trending")
        for i in range(5):
            ce.update(92.0 - i, 90.0 - i, 91.0 - i, atr=3.0, side=-1)
        trough_stop = ce.get_stop()
        for i in range(5):
            ce.update(90.0 + i, 88.0 + i, 89.0 + i, atr=3.0, side=-1)
            assert ce.get_stop() <= trough_stop

    def test_is_stopped_short(self):
        """Short position stopped when close >= stop."""
        ce = ChandelierExit(atr_mult=2.0, regime="trending")
        ce.update(100.0, 95.0, 97.0, atr=3.0, side=-1)
        stop = ce.get_stop()
        assert not ce.is_stopped(stop - 0.01, -1)
        assert ce.is_stopped(stop, -1)
        assert ce.is_stopped(stop + 1.0, -1)


class TestChandelierExitReset:
    """Reset behaviour."""

    def test_reset_clears_state(self):
        """After reset, stop should be NaN."""
        ce = ChandelierExit(atr_mult=2.0)
        ce.update(100.0, 98.0, 99.0, atr=3.0, side=1)
        assert not np.isnan(ce.get_stop())
        ce.reset()
        assert np.isnan(ce.get_stop())

    def test_new_trade_after_reset(self):
        """After reset, a new trade should start fresh."""
        ce = ChandelierExit(atr_mult=2.0, regime="strong_trend")
        ce.update(100.0, 98.0, 99.0, atr=3.0, side=1)
        first_stop = ce.get_stop()
        ce.reset()
        ce.update(200.0, 198.0, 199.0, atr=3.0, side=1)
        assert ce.get_stop() != first_stop
        assert ce.get_stop() == pytest.approx(200.0 - 2.0 * 3.0)


class TestChandelierExitMeanReversion:
    """Mean-reversion regime uses entry price, not extremum."""

    def test_uses_entry_price_long(self):
        """Mean-reversion long stop based on entry, not highest."""
        ce = ChandelierExit(atr_mult=2.0, regime="mean_reversion")
        ce.update(100.0, 98.0, 99.0, atr=3.0, side=1)
        entry_stop = ce.get_stop()
        # Price goes much higher – stop should NOT move up (based on entry).
        ce.update(120.0, 118.0, 119.0, atr=3.0, side=1)
        assert ce.get_stop() == pytest.approx(entry_stop)

    def test_uses_entry_price_short(self):
        """Mean-reversion short stop based on entry, not lowest."""
        ce = ChandelierExit(atr_mult=2.0, regime="mean_reversion")
        ce.update(100.0, 98.0, 99.0, atr=3.0, side=-1)
        entry_stop = ce.get_stop()
        ce.update(80.0, 78.0, 79.0, atr=3.0, side=-1)
        assert ce.get_stop() == pytest.approx(entry_stop)


class TestChandelierExitCrisis:
    """Crisis regime time-stop tests."""

    def test_crisis_time_stop_triggers(self):
        """Crisis: N bars without profit -> forced exit."""
        ce = ChandelierExit(atr_mult=1.5, regime="crisis", crisis_time_stop=5)
        entry = 100.0
        # 5 bars of no profit (close at or below entry for long).
        for i in range(5):
            ce.update(101.0, 98.0, entry - 0.5, atr=2.0, side=1)
        assert ce.is_stopped(entry - 0.5, 1)

    def test_crisis_time_stop_no_trigger_if_profitable(self):
        """Crisis: should NOT trigger if trade was profitable at some point."""
        ce = ChandelierExit(atr_mult=1.5, regime="crisis", crisis_time_stop=5)
        ce.update(100.0, 98.0, 99.0, atr=2.0, side=1)
        # Make it profitable.
        ce.update(105.0, 103.0, 104.0, atr=2.0, side=1)
        # Then give back.
        for _ in range(5):
            ce.update(100.0, 98.0, 98.5, atr=2.0, side=1)
        # best_pnl > 0 so time stop should NOT fire.
        # (price stop may or may not fire – check only time-stop logic)
        # Use a wide enough stop that price stop doesn't trigger.
        ce2 = ChandelierExit(atr_mult=100.0, regime="crisis", crisis_time_stop=5)
        ce2.update(100.0, 98.0, 99.0, atr=2.0, side=1)
        ce2.update(105.0, 103.0, 104.0, atr=2.0, side=1)
        for _ in range(5):
            ce2.update(100.0, 98.0, 98.5, atr=2.0, side=1)
        assert not ce2.is_stopped(98.5, 1)

    def test_crisis_time_stop_not_before_n_bars(self):
        """Crisis time stop should not trigger before N bars."""
        ce = ChandelierExit(atr_mult=100.0, regime="crisis", crisis_time_stop=10)
        for _ in range(9):
            ce.update(100.0, 98.0, 98.5, atr=2.0, side=1)
        assert not ce.is_stopped(98.5, 1)


class TestChandelierExitFlat:
    """Behaviour when side=0."""

    def test_flat_no_update(self):
        """Side=0 should not change state."""
        ce = ChandelierExit(atr_mult=2.0)
        ce.update(100.0, 98.0, 99.0, atr=3.0, side=0)
        assert np.isnan(ce.get_stop())

    def test_is_stopped_flat(self):
        """is_stopped returns False when flat."""
        ce = ChandelierExit(atr_mult=2.0)
        assert not ce.is_stopped(50.0, 0)


class TestChandelierExitVectorised:
    """Vectorised compute_stops."""

    def test_basic_long_vectorised(self):
        """Vectorised long stops should match iterative."""
        n = 20
        highs = 100.0 + np.arange(n, dtype=float)
        lows = 98.0 + np.arange(n, dtype=float)
        closes = 99.0 + np.arange(n, dtype=float)
        atrs = np.full(n, 3.0)
        entries = np.full(n, 99.0)
        sides = np.ones(n)
        stops = ChandelierExit.compute_stops(highs, lows, closes, atrs, entries, sides, 2.0)
        # All stops should be non-NaN and non-decreasing.
        for i in range(1, n):
            assert stops[i] >= stops[i - 1]

    def test_flat_bars_produce_nan(self):
        """Flat bars in vectorised mode should produce NaN stops."""
        n = 5
        sides = np.array([0, 1, 1, 0, 0], dtype=float)
        stops = ChandelierExit.compute_stops(
            np.full(n, 100.0), np.full(n, 98.0), np.full(n, 99.0),
            np.full(n, 2.0), np.full(n, 99.0), sides, 2.0,
        )
        assert np.isnan(stops[0])
        assert not np.isnan(stops[1])
        assert np.isnan(stops[3])


class TestChandelierExitDefaults:
    """Default ATR multipliers per regime."""

    def test_default_multiplier_strong_trend(self):
        ce = ChandelierExit(regime="strong_trend")
        ce.update(100.0, 98.0, 99.0, atr=2.0, side=1)
        assert ce.get_stop() == pytest.approx(100.0 - 3.0 * 2.0)

    def test_default_multiplier_mild_trend(self):
        ce = ChandelierExit(regime="mild_trend")
        ce.update(100.0, 98.0, 99.0, atr=2.0, side=1)
        assert ce.get_stop() == pytest.approx(100.0 - 2.25 * 2.0)


# ======================================================================
# VolTargeting
# ======================================================================


class TestRealizedVol:
    """Tests for realized_vol."""

    def test_zero_returns_zero_vol(self):
        """Constant returns should produce near-zero vol."""
        rets = np.zeros(100)
        rv = realized_vol(rets, halflife=20)
        assert rv[-1] == pytest.approx(0.0, abs=1e-10)

    def test_high_vol_returns(self):
        """Alternating +/- returns should produce high vol."""
        rets = np.array([0.05, -0.05] * 50)
        rv = realized_vol(rets, halflife=20)
        assert rv[-1] > 0.3  # Should be meaningfully above zero.

    def test_output_length(self):
        """Output length matches input."""
        rets = np.random.randn(200) * 0.01
        rv = realized_vol(rets, halflife=60)
        assert len(rv) == 200

    def test_annualised(self):
        """Vol should be annualised (scaled by sqrt(252))."""
        rets = np.random.randn(500) * 0.01
        rv = realized_vol(rets, halflife=60)
        # Daily std ~ 0.01, annualised ~ 0.01 * sqrt(252) ~ 0.159
        assert 0.05 < rv[-1] < 0.40


class TestVolScale:
    """Tests for vol_scale."""

    def test_high_vol_low_scale(self):
        """High realised vol -> low position scale."""
        rv = np.array([0.30])  # 30% annualised
        vs = vol_scale(0.10, rv)
        assert vs[0] == pytest.approx(0.10 / 0.30, rel=1e-6)

    def test_low_vol_high_scale(self):
        """Low realised vol -> high position scale."""
        rv = np.array([0.05])
        vs = vol_scale(0.10, rv)
        assert vs[0] == pytest.approx(2.0)

    def test_clip_upper(self):
        """Scale should be clipped at 3.0."""
        rv = np.array([0.01])  # Very low vol
        vs = vol_scale(0.10, rv)
        assert vs[0] == 3.0

    def test_clip_lower(self):
        """Scale should be clipped at 0.2."""
        rv = np.array([1.0])  # Very high vol
        vs = vol_scale(0.10, rv)
        assert vs[0] == 0.2

    def test_zero_vol_defaults_to_one(self):
        """Zero realised vol should produce scale = 1.0 (clipped)."""
        rv = np.array([0.0])
        vs = vol_scale(0.10, rv)
        assert vs[0] == 1.0


class TestAtrPercentile:
    """Tests for atr_percentile."""

    def test_monotonic_increasing(self):
        """Monotonically increasing ATR -> last bar at 100th percentile."""
        atr = np.arange(1.0, 101.0)
        pctl = atr_percentile(atr, lookback=100)
        assert pctl[-1] == pytest.approx(100.0)

    def test_monotonic_decreasing(self):
        """Monotonically decreasing ATR -> last bar at low percentile."""
        atr = np.arange(100.0, 0.0, -1.0)
        pctl = atr_percentile(atr, lookback=100)
        assert pctl[-1] < 5.0

    def test_constant_atr(self):
        """Constant ATR -> 100th percentile (all values equal)."""
        atr = np.full(50, 5.0)
        pctl = atr_percentile(atr, lookback=50)
        assert pctl[-1] == pytest.approx(100.0)

    def test_all_values_in_range(self):
        """All percentile values should be in [0, 100]."""
        rng = np.random.RandomState(42)
        atr = np.abs(rng.randn(300)) * 2.0 + 1.0
        pctl = atr_percentile(atr, lookback=252)
        valid = pctl[~np.isnan(pctl)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_output_length_matches_input(self):
        """Output array length equals input array length."""
        atr = np.arange(1.0, 51.0)
        pctl = atr_percentile(atr, lookback=20)
        assert len(pctl) == len(atr)

    def test_single_bar_returns_50(self):
        """Single bar window returns 50.0 (default)."""
        atr = np.array([3.0])
        pctl = atr_percentile(atr, lookback=252)
        assert pctl[0] == pytest.approx(50.0)


class TestExtremeVolAdj:
    """Tests for extreme_vol_adjustment."""

    def test_above_90(self):
        assert extreme_vol_adjustment(np.array([95.0]))[0] == 0.5

    def test_between_80_90(self):
        assert extreme_vol_adjustment(np.array([85.0]))[0] == 0.75

    def test_at_80(self):
        """Exactly 80 -> not extreme (<=80 is normal)."""
        assert extreme_vol_adjustment(np.array([80.0]))[0] == 1.0

    def test_at_90(self):
        """Exactly 90 -> 0.75 band (>80 and <=90)."""
        assert extreme_vol_adjustment(np.array([90.0]))[0] == 0.75

    def test_below_80(self):
        assert extreme_vol_adjustment(np.array([50.0]))[0] == 1.0


class TestVolTargetingClass:
    """Tests for VolTargeting class."""

    def test_compute_returns_array(self):
        vt = VolTargeting(target_vol=0.10, halflife=20, atr_lookback=50)
        rets = np.random.randn(100) * 0.01
        atr = np.abs(np.random.randn(100)) * 2.0
        result = vt.compute(rets, atr)
        assert len(result) == 100
        assert np.all(np.isfinite(result) | np.isnan(result))


# ======================================================================
# PositionSizer
# ======================================================================


class TestCalcLots:
    """Tests for calc_lots."""

    def test_basic_sizing(self):
        """Standard sizing: (1M * 0.02) / (50 * 10) = 40 lots."""
        lots = calc_lots(
            equity=1_000_000, risk_pct=0.02, stop_distance=50.0,
            multiplier=10, price=4000.0, margin_rate=0.12,
        )
        assert lots == 40

    def test_margin_constraint(self):
        """Margin should cap the number of lots."""
        lots = calc_lots(
            equity=100_000, risk_pct=0.02, stop_distance=10.0,
            multiplier=10, price=5000.0, margin_rate=0.15,
        )
        risk_lots = (100_000 * 0.02) / (10.0 * 10)  # 20
        margin_lots = (100_000 * 0.30) / (5000.0 * 10 * 0.15)  # 4
        assert lots == int(math.floor(min(risk_lots, margin_lots)))

    def test_minimum_one_lot(self):
        """Should always return at least 1 lot."""
        lots = calc_lots(
            equity=1000, risk_pct=0.001, stop_distance=1000.0,
            multiplier=100, price=5000.0, margin_rate=0.50,
        )
        assert lots == 1

    def test_zero_stop_distance(self):
        """Zero stop distance should return 1 (guard)."""
        lots = calc_lots(
            equity=1_000_000, risk_pct=0.02, stop_distance=0.0,
            multiplier=10, price=4000.0, margin_rate=0.12,
        )
        assert lots == 1

    def test_large_equity(self):
        """Large equity should produce more lots."""
        small = calc_lots(
            equity=500_000, risk_pct=0.02, stop_distance=50.0,
            multiplier=10, price=4000.0, margin_rate=0.12,
        )
        large = calc_lots(
            equity=5_000_000, risk_pct=0.02, stop_distance=50.0,
            multiplier=10, price=4000.0, margin_rate=0.12,
        )
        assert large > small

    def test_zero_equity(self):
        """Zero equity -> 1 lot (guard)."""
        assert calc_lots(0, 0.02, 50.0, 10, 4000.0, 0.12) == 1


class TestPositionSizerClass:
    """Tests for PositionSizer class."""

    def test_uses_config_defaults(self):
        ps = PositionSizer()
        lots = ps.size(equity=1_000_000, stop_distance=50.0, multiplier=10, price=4000.0, margin_rate=0.12)
        assert lots >= 1

    def test_override_risk_pct(self):
        ps = PositionSizer(risk_pct=0.01)
        lots = ps.size(equity=1_000_000, stop_distance=50.0, multiplier=10, price=4000.0, margin_rate=0.12)
        expected = calc_lots(1_000_000, 0.01, 50.0, 10, 4000.0, 0.12, 0.30)
        assert lots == expected


# ======================================================================
# DirectionalFilter
# ======================================================================


class TestFilterSignal:
    """Tests for filter_signal function."""

    def test_long_filters_negative(self):
        assert filter_signal(-0.5, "long") == 0.0

    def test_long_passes_positive(self):
        assert filter_signal(0.8, "long") == 0.8

    def test_long_passes_zero(self):
        assert filter_signal(0.0, "long") == 0.0

    def test_short_filters_positive(self):
        assert filter_signal(0.5, "short") == 0.0

    def test_short_passes_negative(self):
        assert filter_signal(-0.8, "short") == -0.8

    def test_short_passes_zero(self):
        assert filter_signal(0.0, "short") == 0.0

    def test_neutral_passes_positive(self):
        assert filter_signal(0.5, "neutral") == 0.5

    def test_neutral_passes_negative(self):
        assert filter_signal(-0.5, "neutral") == -0.5


class TestLoadDirection:
    """Tests for load_direction."""

    def test_loads_known_instrument(self):
        d = load_direction("RB")
        assert d in ("long", "short", "neutral")

    def test_unknown_instrument_raises(self):
        with pytest.raises(KeyError):
            load_direction("UNKNOWN_INSTRUMENT_XYZ")


class TestDirectionalFilterClass:
    """Tests for DirectionalFilter class."""

    def test_apply_neutral(self):
        df = DirectionalFilter("RB")
        # RB default is neutral.
        assert df.apply(-0.5) == -0.5
        assert df.apply(0.5) == 0.5

    def test_direction_property(self):
        df = DirectionalFilter("RB")
        assert df.direction in ("long", "short", "neutral")


# ======================================================================
# VolClassifier
# ======================================================================


class TestClassifyVol:
    """Tests for classify_vol."""

    def test_high(self):
        assert classify_vol(75.0) == "high"

    def test_low(self):
        assert classify_vol(25.0) == "low"

    def test_mid(self):
        assert classify_vol(50.0) == "mid"

    def test_boundary_70(self):
        """Exactly 70 -> mid (not >70)."""
        assert classify_vol(70.0) == "mid"

    def test_boundary_30(self):
        """Exactly 30 -> mid (not <30)."""
        assert classify_vol(30.0) == "mid"

    def test_boundary_70_01(self):
        assert classify_vol(70.01) == "high"

    def test_boundary_29_99(self):
        assert classify_vol(29.99) == "low"

    def test_zero(self):
        assert classify_vol(0.0) == "low"

    def test_hundred(self):
        assert classify_vol(100.0) == "high"


# ======================================================================
# PortfolioStops
# ======================================================================


class TestPortfolioStops:
    """Tests for PortfolioStops."""

    def test_normal(self):
        ps = PortfolioStops(warning=-0.10, reduce=-0.15, circuit=-0.20, daily=-0.05)
        assert ps.check(-0.05, -0.01) == "normal"

    def test_warning(self):
        ps = PortfolioStops(warning=-0.10, reduce=-0.15, circuit=-0.20, daily=-0.05)
        assert ps.check(-0.10, -0.01) == "warning"

    def test_reduce(self):
        ps = PortfolioStops(warning=-0.10, reduce=-0.15, circuit=-0.20, daily=-0.05)
        assert ps.check(-0.15, -0.01) == "reduce"

    def test_circuit(self):
        ps = PortfolioStops(warning=-0.10, reduce=-0.15, circuit=-0.20, daily=-0.05)
        assert ps.check(-0.20, -0.01) == "circuit"

    def test_daily_circuit(self):
        ps = PortfolioStops(warning=-0.10, reduce=-0.15, circuit=-0.20, daily=-0.05)
        assert ps.check(-0.05, -0.05) == "daily_circuit"

    def test_daily_circuit_priority(self):
        """Daily circuit takes priority over drawdown levels."""
        ps = PortfolioStops(warning=-0.10, reduce=-0.15, circuit=-0.20, daily=-0.05)
        assert ps.check(-0.25, -0.06) == "daily_circuit"

    def test_deep_drawdown(self):
        """Drawdown beyond circuit threshold."""
        ps = PortfolioStops(warning=-0.10, reduce=-0.15, circuit=-0.20, daily=-0.05)
        assert ps.check(-0.30, -0.01) == "circuit"

    def test_multiplier_normal(self):
        assert PortfolioStops.get_position_multiplier("normal") == 1.0

    def test_multiplier_warning(self):
        assert PortfolioStops.get_position_multiplier("warning") == 1.0

    def test_multiplier_reduce(self):
        assert PortfolioStops.get_position_multiplier("reduce") == 0.5

    def test_multiplier_circuit(self):
        assert PortfolioStops.get_position_multiplier("circuit") == 0.0

    def test_multiplier_daily_circuit(self):
        assert PortfolioStops.get_position_multiplier("daily_circuit") == 0.0

    def test_multiplier_unknown_defaults_to_one(self):
        assert PortfolioStops.get_position_multiplier("unknown") == 1.0

    def test_config_defaults(self):
        """PortfolioStops should load from config when no args given."""
        ps = PortfolioStops()
        # Should not raise. Just check it works.
        result = ps.check(-0.05, -0.01)
        assert result in ("normal", "warning", "reduce", "circuit", "daily_circuit")

    def test_between_warning_and_reduce(self):
        ps = PortfolioStops(warning=-0.10, reduce=-0.15, circuit=-0.20, daily=-0.05)
        assert ps.check(-0.12, -0.01) == "warning"

    def test_between_reduce_and_circuit(self):
        ps = PortfolioStops(warning=-0.10, reduce=-0.15, circuit=-0.20, daily=-0.05)
        assert ps.check(-0.17, -0.01) == "reduce"
