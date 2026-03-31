"""Tests for config module."""

import pytest
from pathlib import Path
from config import (
    get_settings,
    get_fundamental_views,
    get_regime_thresholds,
    get_alphaforge_path,
    get_data_dir,
    get_instruments,
    get_frequencies,
    clear_cache,
)


@pytest.fixture(autouse=True)
def _clear():
    clear_cache()
    yield
    clear_cache()


class TestSettings:
    def test_loads_settings(self):
        s = get_settings()
        assert isinstance(s, dict)
        assert "initial_capital" in s
        assert "alphaforge_path" in s

    def test_capital_is_positive(self):
        s = get_settings()
        assert s["initial_capital"] > 0

    def test_risk_per_trade_range(self):
        s = get_settings()
        assert 0 < s["risk_per_trade"] <= 0.05

    def test_vol_target_range(self):
        s = get_settings()
        assert 0 < s["target_vol"] <= 0.30

    def test_stop_levels_ordered(self):
        s = get_settings()
        assert s["stop_warning"] > s["stop_reduce"] > s["stop_circuit"]

    def test_instruments_list(self):
        instruments = get_instruments()
        assert isinstance(instruments, list)
        assert len(instruments) >= 5
        assert "RB" in instruments
        assert "I" in instruments

    def test_frequencies_list(self):
        freqs = get_frequencies()
        assert "1h" in freqs
        assert "daily" in freqs


class TestFundamentalViews:
    def test_loads_views(self):
        v = get_fundamental_views()
        assert "views" in v
        assert "updated_at" in v

    def test_all_instruments_have_views(self):
        v = get_fundamental_views()
        instruments = get_instruments()
        for inst in instruments:
            assert inst in v["views"], f"Missing view for {inst}"

    def test_direction_values(self):
        v = get_fundamental_views()
        valid_directions = {"long", "short", "neutral"}
        for inst, view in v["views"].items():
            assert view["direction"] in valid_directions, f"Invalid direction for {inst}: {view['direction']}"


class TestRegimeThresholds:
    def test_loads_defaults(self):
        t = get_regime_thresholds()
        assert "strong_trend_pct" in t
        assert "mild_trend_pct" in t
        assert "buffer_months" in t

    def test_thresholds_ordered(self):
        t = get_regime_thresholds()
        assert t["mild_trend_pct"] < t["strong_trend_pct"]

    def test_buffer_positive(self):
        t = get_regime_thresholds()
        assert t["buffer_months"] > 0

    def test_instrument_override(self):
        t_default = get_regime_thresholds()
        t_iron = get_regime_thresholds("I")
        assert t_iron["strong_trend_pct"] > t_default["strong_trend_pct"]

    def test_unknown_instrument_uses_defaults(self):
        t = get_regime_thresholds("UNKNOWN")
        t_default = get_regime_thresholds()
        assert t == t_default

    def test_crisis_sigma_positive(self):
        t = get_regime_thresholds()
        assert t["crisis_atr_sigma"] > 0


class TestPaths:
    def test_alphaforge_path_is_path(self):
        p = get_alphaforge_path()
        assert isinstance(p, Path)

    def test_data_dir_is_path(self):
        p = get_data_dir()
        assert isinstance(p, Path)


class TestCaching:
    def test_cache_returns_same_object(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_clear_cache_reloads(self):
        s1 = get_settings()
        clear_cache()
        s2 = get_settings()
        assert s1 is not s2
        assert s1 == s2
