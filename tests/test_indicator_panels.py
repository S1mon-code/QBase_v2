"""Tests for indicator panel packaging (AlphaForge INDICATOR_PANEL_SPEC).

Covers:
  - Base class: get_indicator_panels returns empty panels by default
  - Helper methods: _make_overlay, _make_subplot, _make_subplot_trace
  - Strategy overrides: correct overlay/subplot classification
  - backtest_runner: _inject_indicator_panels packages data correctly
  - Data format: conforms to INDICATOR_PANEL_SPEC structure
"""

from __future__ import annotations

import importlib
import numpy as np
import pytest

from strategies.templates.base_strategy import QBaseStrategy
from strategies.templates.trending_template import TrendingStrategy


def _load_strategy_class(module_path: str):
    """Import a strategy module and return the first concrete strategy class."""
    mod = importlib.import_module(module_path)
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "regime")
            and hasattr(obj, "_generate_signal")
            and obj.__name__ not in (
                "QBaseStrategy", "TrendingStrategy", "MeanReversionStrategy",
            )
        ):
            return obj
    raise ImportError(f"No strategy class found in {module_path}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_arrays(
    n: int = 200,
    base_price: float = 800.0,
    step: float = 2.0,
) -> dict[str, np.ndarray]:
    """Generate synthetic OHLCV + OI arrays."""
    rng = np.random.RandomState(42)
    closes = base_price + np.arange(n, dtype=np.float64) * step
    noise = rng.uniform(0, step * 0.3, size=n)
    highs = closes + noise + step * 0.5
    lows = closes - noise - step * 0.5
    opens = closes - rng.uniform(-step * 0.2, step * 0.2, size=n)
    volumes = rng.uniform(5000, 50000, size=n).astype(np.float64)
    oi = rng.uniform(100000, 200000, size=n).astype(np.float64)
    datetimes = np.arange(n, dtype=np.float64)
    return {
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "opens": opens,
        "volumes": volumes,
        "oi": oi,
        "datetimes": datetimes,
    }


def _init_strategy(strategy: QBaseStrategy, arrays: dict) -> None:
    """Call on_init_arrays on a strategy."""
    strategy.on_init_arrays(
        closes=arrays["closes"],
        highs=arrays["highs"],
        lows=arrays["lows"],
        opens=arrays["opens"],
        volumes=arrays["volumes"],
        oi=arrays["oi"],
        datetimes=arrays["datetimes"],
    )


# ---------------------------------------------------------------------------
# Base class tests
# ---------------------------------------------------------------------------

class TestBaseIndicatorPanels:
    """Test QBaseStrategy indicator panel base functionality."""

    def test_default_returns_empty_panels(self):
        """Base get_indicator_panels returns empty overlays and subplots."""

        class _Dummy(TrendingStrategy):
            name = "dummy"
            regime = "trending"
            horizon = "fast"
            direction = "long"
            signal_dimensions = ["momentum"]
            warmup = 10

            def _generate_signal(self, bar_index):
                return 0.0

        s = _Dummy()
        arrays = _make_arrays()
        _init_strategy(s, arrays)
        panels = s.get_indicator_panels(arrays["datetimes"])

        assert isinstance(panels, dict)
        assert "overlays" in panels
        assert "subplots" in panels
        assert panels["overlays"] == []
        assert panels["subplots"] == []

    def test_auto_classify_overlay(self):
        """Auto-classify EMA as overlay from get_indicator_config with array."""

        class _Auto(TrendingStrategy):
            name = "auto_test"
            regime = "trending"
            horizon = "fast"
            direction = "long"
            signal_dimensions = ["momentum"]
            warmup = 10

            def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
                super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
                self._my_ema = closes * 0.99  # fake EMA
                self._my_rsi = np.full(len(closes), 50.0)  # fake RSI

            def _generate_signal(self, bar_index):
                return 0.0

            def get_indicator_config(self):
                return [
                    {"name": "EMA(20)", "array": self._my_ema},
                    {"name": "RSI(14)", "array": self._my_rsi},
                ]

        s = _Auto()
        arrays = _make_arrays()
        _init_strategy(s, arrays)
        panels = s.get_indicator_panels(arrays["datetimes"])

        # EMA auto-classified as overlay
        assert len(panels["overlays"]) == 1
        assert "EMA" in panels["overlays"][0]["name"]

        # RSI auto-classified as subplot with hints
        assert len(panels["subplots"]) == 1
        rsi_panel = panels["subplots"][0]
        assert "RSI" in rsi_panel["name"]
        assert rsi_panel.get("y_range") == [0, 100]
        assert rsi_panel.get("horizontal_lines") == [30, 70]

    def test_auto_classify_unknown_indicator(self):
        """Unknown indicator name defaults to subplot."""

        class _Unknown(TrendingStrategy):
            name = "unknown_test"
            regime = "trending"
            horizon = "fast"
            direction = "long"
            signal_dimensions = ["momentum"]
            warmup = 10

            def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
                super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
                self._my_thing = closes * 1.0

            def _generate_signal(self, bar_index):
                return 0.0

            def get_indicator_config(self):
                return [
                    {"name": "MyCoolNewIndicator", "array": self._my_thing},
                ]

        s = _Unknown()
        arrays = _make_arrays()
        _init_strategy(s, arrays)
        panels = s.get_indicator_panels(arrays["datetimes"])

        # Unknown → subplot (safe default)
        assert len(panels["overlays"]) == 0
        assert len(panels["subplots"]) == 1

    def test_panel_grouping(self):
        """Traces with same 'panel' name are grouped into one subplot."""

        class _Grouped(TrendingStrategy):
            name = "grouped_test"
            regime = "trending"
            horizon = "fast"
            direction = "long"
            signal_dimensions = ["momentum"]
            warmup = 10

            def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
                super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
                self._line = closes * 0.01
                self._signal = closes * 0.005
                self._hist = closes * 0.002

            def _generate_signal(self, bar_index):
                return 0.0

            def get_indicator_config(self):
                return [
                    {"name": "MACD Line", "array": self._line, "panel": "MACD"},
                    {"name": "Signal", "array": self._signal, "panel": "MACD"},
                    {"name": "Hist", "array": self._hist, "panel": "MACD",
                     "style": "bar", "color_positive": "#26a69a", "color_negative": "#ef5350"},
                ]

        s = _Grouped()
        arrays = _make_arrays()
        _init_strategy(s, arrays)
        panels = s.get_indicator_panels(arrays["datetimes"])

        # All 3 traces grouped into 1 MACD panel
        assert len(panels["subplots"]) == 1
        macd_panel = panels["subplots"][0]
        assert macd_panel["name"] == "MACD"
        assert len(macd_panel["traces"]) == 3

    def test_legacy_config_no_array(self):
        """Legacy get_indicator_config without 'array' returns empty panels."""

        class _Legacy(TrendingStrategy):
            name = "legacy_test"
            regime = "trending"
            horizon = "fast"
            direction = "long"
            signal_dimensions = ["momentum"]
            warmup = 10

            def _generate_signal(self, bar_index):
                return 0.0

            def get_indicator_config(self):
                return [
                    {"name": "ema", "params": {"period": 20}},
                ]

        s = _Legacy()
        arrays = _make_arrays()
        _init_strategy(s, arrays)
        panels = s.get_indicator_panels(arrays["datetimes"])

        assert panels["overlays"] == []
        assert panels["subplots"] == []

    def test_make_overlay(self):
        """_make_overlay builds correct overlay dict."""
        dt = np.arange(5, dtype=np.float64)
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        ov = QBaseStrategy._make_overlay("EMA(20)", dt, data, style="line", color="#ffab40")

        assert ov["name"] == "EMA(20)"
        assert ov["style"] == "line"
        assert ov["color"] == "#ffab40"
        assert len(ov["data"]) == 5
        # NaN values should be converted to None
        assert ov["data"][2][1] is None
        assert ov["data"][0][1] == 1.0

    def test_make_overlay_no_color(self):
        """_make_overlay without color omits the key."""
        dt = np.arange(3, dtype=np.float64)
        data = np.array([1.0, 2.0, 3.0])

        ov = QBaseStrategy._make_overlay("Test", dt, data)

        assert "color" not in ov
        assert ov["style"] == "line"

    def test_make_subplot_trace(self):
        """_make_subplot_trace builds correct trace dict."""
        dt = np.arange(3, dtype=np.float64)
        data = np.array([-1.0, 0.0, 1.0])

        tr = QBaseStrategy._make_subplot_trace(
            "MACD Line", dt, data, style="line", color="#4fc3f7",
        )

        assert tr["name"] == "MACD Line"
        assert tr["style"] == "line"
        assert tr["color"] == "#4fc3f7"
        assert len(tr["data"]) == 3

    def test_make_subplot_trace_bar_colors(self):
        """_make_subplot_trace with bar style supports pos/neg colors."""
        dt = np.arange(3, dtype=np.float64)
        data = np.array([-1.0, 0.0, 1.0])

        tr = QBaseStrategy._make_subplot_trace(
            "Histogram", dt, data, style="bar",
            color_positive="#26a69a", color_negative="#ef5350",
        )

        assert tr["style"] == "bar"
        assert tr["color_positive"] == "#26a69a"
        assert tr["color_negative"] == "#ef5350"
        assert "color" not in tr

    def test_make_subplot(self):
        """_make_subplot builds correct panel dict."""
        traces = [{"name": "RSI", "data": [], "style": "line"}]

        panel = QBaseStrategy._make_subplot(
            "RSI(14)", traces, height_ratio=0.12,
            horizontal_lines=[30, 70], y_range=[0, 100],
        )

        assert panel["name"] == "RSI(14)"
        assert panel["height_ratio"] == 0.12
        assert panel["horizontal_lines"] == [30, 70]
        assert panel["y_range"] == [0, 100]
        assert "zero_line" not in panel
        assert panel["traces"] == traces

    def test_make_subplot_with_zero_line(self):
        """_make_subplot with zero_line=True includes it."""
        traces = [{"name": "MACD", "data": [], "style": "line"}]

        panel = QBaseStrategy._make_subplot("MACD", traces, zero_line=True)

        assert panel["zero_line"] is True


# ---------------------------------------------------------------------------
# Strategy override tests (sample a few strategies)
# ---------------------------------------------------------------------------

class TestStrategyPanels:
    """Test that individual strategies return correct panel classifications."""

    @pytest.fixture
    def arrays(self):
        return _make_arrays(n=200)

    def test_v1_overlay_and_subplot(self, arrays):
        """v1 has EMA overlays + RSI subplot."""
        V1Class = _load_strategy_class("strategies.strong_trend.long.I.1h.v1")

        s = V1Class()
        _init_strategy(s, arrays)
        panels = s.get_indicator_panels(arrays["datetimes"])

        assert len(panels["overlays"]) == 2  # EMA fast + EMA slow
        assert len(panels["subplots"]) >= 1  # RSI

        # Check overlay names contain EMA
        overlay_names = [o["name"] for o in panels["overlays"]]
        assert any("EMA" in n for n in overlay_names)

        # Check RSI subplot
        rsi_panel = panels["subplots"][0]
        assert "RSI" in rsi_panel["name"]
        assert rsi_panel.get("y_range") == [0, 100]

    def test_v3_pure_subplots(self, arrays):
        """v3 (MACD + CMF) has no overlays, only subplots."""
        V3Class = _load_strategy_class("strategies.strong_trend.long.I.1h.v3")

        s = V3Class()
        _init_strategy(s, arrays)
        panels = s.get_indicator_panels(arrays["datetimes"])

        assert len(panels["overlays"]) == 0
        assert len(panels["subplots"]) >= 2  # MACD + CMF

        # MACD panel should have 3 traces (line, signal, histogram)
        macd_panel = panels["subplots"][0]
        assert "MACD" in macd_panel["name"]
        assert len(macd_panel["traces"]) == 3

    def test_v7_bollinger_overlay(self, arrays):
        """v7 has Bollinger Bands overlays."""
        V7Class = _load_strategy_class("strategies.strong_trend.long.I.1h.v7")

        s = V7Class()
        _init_strategy(s, arrays)
        panels = s.get_indicator_panels(arrays["datetimes"])

        assert len(panels["overlays"]) == 3  # Upper, Middle, Lower
        # Check dash style for bands
        styles = {o["style"] for o in panels["overlays"]}
        assert "dash" in styles


# ---------------------------------------------------------------------------
# Data format compliance tests
# ---------------------------------------------------------------------------

class TestPanelDataFormat:
    """Verify panels conform to INDICATOR_PANEL_SPEC format."""

    @pytest.fixture
    def sample_panels(self):
        """Get panels from v1 strategy as a representative sample."""
        arrays = _make_arrays(n=200)
        V1Class = _load_strategy_class("strategies.strong_trend.long.I.1h.v1")

        s = V1Class()
        _init_strategy(s, arrays)
        return s.get_indicator_panels(arrays["datetimes"])

    def test_overlay_structure(self, sample_panels):
        """Each overlay has required fields: name, data, style."""
        for ov in sample_panels["overlays"]:
            assert "name" in ov
            assert "data" in ov
            assert "style" in ov
            assert ov["style"] in ("line", "step", "dash")
            # Data is list of (datetime, value) tuples
            assert isinstance(ov["data"], list)
            if len(ov["data"]) > 0:
                assert len(ov["data"][0]) == 2

    def test_subplot_structure(self, sample_panels):
        """Each subplot has required fields: name, height_ratio, traces."""
        for sp in sample_panels["subplots"]:
            assert "name" in sp
            assert "height_ratio" in sp
            assert "traces" in sp
            assert 0.05 <= sp["height_ratio"] <= 0.35

    def test_trace_structure(self, sample_panels):
        """Each trace has required fields: name, data, style."""
        for sp in sample_panels["subplots"]:
            for tr in sp["traces"]:
                assert "name" in tr
                assert "data" in tr
                assert "style" in tr
                assert tr["style"] in ("line", "bar", "area", "step", "dash")

    def test_data_length_matches(self, sample_panels):
        """All data arrays should have the same length (200 bars)."""
        expected_len = 200
        for ov in sample_panels["overlays"]:
            assert len(ov["data"]) == expected_len
        for sp in sample_panels["subplots"]:
            for tr in sp["traces"]:
                assert len(tr["data"]) == expected_len


# ---------------------------------------------------------------------------
# Injection tests
# ---------------------------------------------------------------------------

class TestInjectIndicatorPanels:
    """Test _inject_indicator_panels from backtest_runner."""

    def test_injection_adds_metadata(self):
        """_inject_indicator_panels adds indicator_panels to result.metadata."""
        import sys
        sys.path.insert(0, "/Users/simon/Desktop/QBase_v2")
        from pipeline.backtest_runner import _inject_indicator_panels

        arrays = _make_arrays(n=200)

        # Create a dummy strategy
        class _Dummy(TrendingStrategy):
            name = "dummy"
            regime = "trending"
            horizon = "fast"
            direction = "long"
            signal_dimensions = ["momentum"]
            warmup = 10

            def _generate_signal(self, bar_index):
                return 0.0

        s = _Dummy()
        _init_strategy(s, arrays)
        signals = s.generate_signals()

        # Mock result object
        class _Result:
            metadata = {}

        result = _Result()
        _inject_indicator_panels(result, s, signals, arrays["datetimes"])

        assert "indicator_panels" in result.metadata
        panels = result.metadata["indicator_panels"]
        assert "overlays" in panels
        assert "subplots" in panels

        # Signal subplot always appended
        assert len(panels["subplots"]) >= 1
        signal_panel = panels["subplots"][-1]
        assert signal_panel["name"] == "Signal"
        assert signal_panel["y_range"] == [-1, 1]
        assert signal_panel["zero_line"] is True

    def test_injection_with_none_metadata(self):
        """_inject_indicator_panels handles result.metadata=None."""
        from pipeline.backtest_runner import _inject_indicator_panels

        arrays = _make_arrays(n=100)

        class _Dummy(TrendingStrategy):
            name = "dummy"
            regime = "trending"
            horizon = "fast"
            direction = "long"
            signal_dimensions = ["momentum"]
            warmup = 10

            def _generate_signal(self, bar_index):
                return 0.0

        s = _Dummy()
        _init_strategy(s, arrays)
        signals = s.generate_signals()

        class _Result:
            metadata = None

        result = _Result()
        _inject_indicator_panels(result, s, signals, arrays["datetimes"])

        assert result.metadata is not None
        assert "indicator_panels" in result.metadata

    def test_auto_color_assignment(self):
        """Overlays and traces without color get auto-assigned colors."""
        from pipeline.backtest_runner import _inject_indicator_panels

        arrays = _make_arrays(n=200)
        V1Class = _load_strategy_class("strategies.strong_trend.long.I.1h.v1")

        s = V1Class()
        _init_strategy(s, arrays)
        signals = s.generate_signals()

        class _Result:
            metadata = {}

        result = _Result()
        _inject_indicator_panels(result, s, signals, arrays["datetimes"])

        panels = result.metadata["indicator_panels"]
        # All overlays should have a color
        for ov in panels["overlays"]:
            assert "color" in ov
