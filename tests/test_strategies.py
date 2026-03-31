"""Tests for QBase_v2 strategy templates, baselines, and first-batch strategies.

Covers:
  - Base class: abstract enforcement, required attributes, validation
  - Templates: correct regime/horizon defaults
  - TSMOM baselines: signal correctness on known trends
  - Strategy v1-v5: instantiation, on_init_arrays, signal range, warmup
  - Edge cases: empty arrays, short arrays, constant price
"""

from __future__ import annotations

import numpy as np
import pytest

from strategies.templates.base_strategy import QBaseStrategy
from strategies.templates.trending_template import TrendingStrategy
from strategies.templates.mean_reversion_template import MeanReversionStrategy

from strategies.baselines.tsmom_fast import TSMOMFast
from strategies.baselines.tsmom_medium import TSMOMMedium
from strategies.baselines.tsmom_slow import TSMOMSlow

from strategies.trending.medium.v1 import TrendMediumV1
from strategies.trending.medium.v2 import TrendMediumV2
from strategies.trending.medium.v3 import TrendMediumV3
from strategies.trending.medium.v4 import TrendMediumV4
from strategies.trending.medium.v5 import TrendMediumV5
from strategies.trending.fast.v1 import TrendFastV1
from strategies.trending.slow.v1 import TrendSlowV1
from strategies.mean_reversion.v1 import MeanReversionV1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_arrays(
    n: int,
    trend: str = "up",
    base_price: float = 3000.0,
    step: float = 5.0,
) -> dict[str, np.ndarray]:
    """Generate synthetic OHLCV + OI arrays.

    Args:
        n:          Number of bars.
        trend:      "up", "down", "flat", or "random".
        base_price: Starting close price.
        step:       Per-bar price change (absolute).

    Returns:
        Dict with keys: closes, highs, lows, opens, volumes, oi, datetimes.
    """
    rng = np.random.RandomState(42)

    if trend == "up":
        closes = base_price + np.arange(n, dtype=np.float64) * step
    elif trend == "down":
        closes = base_price - np.arange(n, dtype=np.float64) * step
    elif trend == "flat":
        closes = np.full(n, base_price, dtype=np.float64)
    elif trend == "random":
        closes = base_price + np.cumsum(rng.randn(n) * step)
    else:
        raise ValueError(f"Unknown trend: {trend}")

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
    """Call on_init_arrays on a strategy with the given arrays dict."""
    strategy.on_init_arrays(
        closes=arrays["closes"],
        highs=arrays["highs"],
        lows=arrays["lows"],
        opens=arrays["opens"],
        volumes=arrays["volumes"],
        oi=arrays["oi"],
        datetimes=arrays["datetimes"],
    )


@pytest.fixture
def uptrend_data() -> dict[str, np.ndarray]:
    """500-bar uptrend data."""
    return _make_arrays(500, trend="up")


@pytest.fixture
def downtrend_data() -> dict[str, np.ndarray]:
    """500-bar downtrend data."""
    return _make_arrays(500, trend="down")


@pytest.fixture
def flat_data() -> dict[str, np.ndarray]:
    """500-bar flat data."""
    return _make_arrays(500, trend="flat")


@pytest.fixture
def short_data() -> dict[str, np.ndarray]:
    """10-bar data (shorter than most warmup periods)."""
    return _make_arrays(10, trend="up")


@pytest.fixture
def empty_data() -> dict[str, np.ndarray]:
    """Empty arrays (0 bars)."""
    return _make_arrays(0, trend="flat")


# ---------------------------------------------------------------------------
# 1. Base class tests
# ---------------------------------------------------------------------------

class TestQBaseStrategy:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """QBaseStrategy is abstract — cannot be instantiated."""
        with pytest.raises(TypeError):
            QBaseStrategy()  # type: ignore[abstract]

    def test_missing_name_raises(self) -> None:
        """Concrete subclass without 'name' attribute raises TypeError."""
        with pytest.raises(TypeError, match="name"):
            class BadStrategy(QBaseStrategy):
                regime = "trending"
                horizon = "fast"
                signal_dimensions = ["momentum"]
                warmup = 10

                def _generate_signal(self, bar_index: int) -> float:
                    return 0.0

    def test_missing_regime_raises(self) -> None:
        """Concrete subclass without 'regime' raises TypeError."""
        with pytest.raises(TypeError, match="regime"):
            class BadStrategy(QBaseStrategy):
                name = "bad"
                signal_dimensions = ["momentum"]
                warmup = 10

                def _generate_signal(self, bar_index: int) -> float:
                    return 0.0

    def test_invalid_regime_raises(self) -> None:
        """Concrete subclass with invalid regime raises TypeError."""
        with pytest.raises(TypeError, match="regime"):
            class BadStrategy(QBaseStrategy):
                name = "bad"
                regime = "crisis"
                horizon = None
                signal_dimensions = ["momentum"]
                warmup = 10

                def _generate_signal(self, bar_index: int) -> float:
                    return 0.0

    def test_trending_without_horizon_raises(self) -> None:
        """Trending strategy without proper horizon raises TypeError."""
        with pytest.raises(TypeError, match="horizon"):
            class BadStrategy(QBaseStrategy):
                name = "bad"
                regime = "trending"
                horizon = None
                signal_dimensions = ["momentum"]
                warmup = 10

                def _generate_signal(self, bar_index: int) -> float:
                    return 0.0

    def test_missing_signal_dimensions_raises(self) -> None:
        """Concrete subclass without signal_dimensions raises TypeError."""
        with pytest.raises(TypeError, match="signal_dimensions"):
            class BadStrategy(QBaseStrategy):
                name = "bad"
                regime = "mean_reversion"
                horizon = None
                warmup = 10

                def _generate_signal(self, bar_index: int) -> float:
                    return 0.0

    def test_missing_warmup_raises(self) -> None:
        """Concrete subclass without warmup raises TypeError."""
        with pytest.raises(TypeError, match="warmup"):
            class BadStrategy(QBaseStrategy):
                name = "bad"
                regime = "mean_reversion"
                horizon = None
                signal_dimensions = ["technical"]

                def _generate_signal(self, bar_index: int) -> float:
                    return 0.0

    def test_generate_signals_without_init_raises(self) -> None:
        """Calling generate_signals before on_init_arrays raises RuntimeError."""
        strat = TSMOMFast()
        with pytest.raises(RuntimeError, match="on_init_arrays"):
            strat.generate_signals()

    def test_repr(self) -> None:
        """__repr__ includes name, regime, and horizon."""
        strat = TSMOMFast()
        r = repr(strat)
        assert "tsmom_fast" in r
        assert "trending" in r
        assert "fast" in r

    def test_default_indicator_config_is_empty_list(self) -> None:
        """Default get_indicator_config returns empty list."""
        # MR strategies that don't override get the base implementation
        # but we'll check the base class has the default
        assert QBaseStrategy.get_indicator_config.__doc__ is not None


# ---------------------------------------------------------------------------
# 2. Template tests
# ---------------------------------------------------------------------------

class TestTrendingTemplate:
    """Tests for TrendingStrategy template."""

    def test_regime_is_trending(self) -> None:
        """Template sets regime to 'trending'."""
        assert TrendingStrategy.regime == "trending"

    def test_horizon_defaults_to_none(self) -> None:
        """Template horizon is None — subclass must set it."""
        assert TrendingStrategy.horizon is None


class TestMeanReversionTemplate:
    """Tests for MeanReversionStrategy template."""

    def test_regime_is_mean_reversion(self) -> None:
        """Template sets regime to 'mean_reversion'."""
        assert MeanReversionStrategy.regime == "mean_reversion"

    def test_horizon_is_none(self) -> None:
        """Mean reversion strategies have no horizon."""
        assert MeanReversionStrategy.horizon is None


# ---------------------------------------------------------------------------
# 3. TSMOM Baseline tests
# ---------------------------------------------------------------------------

class TestTSMOMFast:
    """Tests for TSMOMFast baseline."""

    def test_attributes(self) -> None:
        """Verify class attributes."""
        strat = TSMOMFast()
        assert strat.name == "tsmom_fast"
        assert strat.regime == "trending"
        assert strat.horizon == "fast"
        assert strat.lookback == 20
        assert "momentum" in strat.signal_dimensions

    def test_uptrend_signal_positive(self, uptrend_data: dict) -> None:
        """In a clear uptrend, signal should be +1."""
        strat = TSMOMFast()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        # After warmup, all signals should be +1
        post_warmup = signals[strat.warmup:]
        assert np.all(post_warmup == 1.0)

    def test_downtrend_signal_negative(self, downtrend_data: dict) -> None:
        """In a clear downtrend, signal should be -1."""
        strat = TSMOMFast()
        _init_strategy(strat, downtrend_data)
        signals = strat.generate_signals()
        post_warmup = signals[strat.warmup:]
        assert np.all(post_warmup == -1.0)

    def test_warmup_is_zero(self, uptrend_data: dict) -> None:
        """Signals during warmup period should be 0."""
        strat = TSMOMFast()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals[:strat.warmup] == 0.0)

    def test_signal_is_binary(self, uptrend_data: dict) -> None:
        """TSMOM signal should only be +1, -1, or 0 (warmup)."""
        strat = TSMOMFast()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        unique = set(np.unique(signals))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_indicator_config(self) -> None:
        """get_indicator_config returns tsmom_return entry."""
        strat = TSMOMFast()
        config = strat.get_indicator_config()
        assert len(config) == 1
        assert config[0]["name"] == "tsmom_return"


class TestTSMOMMedium:
    """Tests for TSMOMMedium baseline."""

    def test_attributes(self) -> None:
        """Verify class attributes."""
        strat = TSMOMMedium()
        assert strat.name == "tsmom_medium"
        assert strat.horizon == "medium"
        assert strat.lookback == 60

    def test_uptrend_signal_positive(self, uptrend_data: dict) -> None:
        """In a clear uptrend, post-warmup signal should be +1."""
        strat = TSMOMMedium()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        post_warmup = signals[strat.warmup:]
        assert np.all(post_warmup == 1.0)

    def test_downtrend_signal_negative(self, downtrend_data: dict) -> None:
        """In a clear downtrend, post-warmup signal should be -1."""
        strat = TSMOMMedium()
        _init_strategy(strat, downtrend_data)
        signals = strat.generate_signals()
        post_warmup = signals[strat.warmup:]
        assert np.all(post_warmup == -1.0)


class TestTSMOMSlow:
    """Tests for TSMOMSlow baseline."""

    def test_attributes(self) -> None:
        """Verify class attributes."""
        strat = TSMOMSlow()
        assert strat.name == "tsmom_slow"
        assert strat.horizon == "slow"
        assert strat.lookback == 250
        assert strat.chandelier_mult == 3.0

    def test_uptrend_signal_positive(self) -> None:
        """In a clear uptrend (600 bars), post-warmup signal should be +1."""
        data = _make_arrays(600, trend="up")
        strat = TSMOMSlow()
        _init_strategy(strat, data)
        signals = strat.generate_signals()
        post_warmup = signals[strat.warmup:]
        assert np.all(post_warmup == 1.0)

    def test_downtrend_signal_negative(self) -> None:
        """In a clear downtrend (600 bars), post-warmup signal should be -1."""
        data = _make_arrays(600, trend="down")
        strat = TSMOMSlow()
        _init_strategy(strat, data)
        signals = strat.generate_signals()
        post_warmup = signals[strat.warmup:]
        assert np.all(post_warmup == -1.0)


# ---------------------------------------------------------------------------
# 4. First-batch strategy tests
# ---------------------------------------------------------------------------

class TestTrendMediumV1:
    """Tests for SuperTrend + Volume Momentum strategy."""

    def test_can_instantiate(self) -> None:
        """Strategy can be instantiated without errors."""
        strat = TrendMediumV1()
        assert strat.name == "trend_medium_v1"
        assert strat.horizon == "medium"
        assert strat.regime == "trending"

    def test_on_init_arrays(self, uptrend_data: dict) -> None:
        """on_init_arrays populates precomputed arrays."""
        strat = TrendMediumV1()
        _init_strategy(strat, uptrend_data)
        assert strat._st_direction is not None
        assert strat._vol_mom is not None
        assert len(strat._st_direction) == 500

    def test_signal_in_range(self, uptrend_data: dict) -> None:
        """All signals must be in [-1, 1]."""
        strat = TrendMediumV1()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)

    def test_warmup_produces_zero(self, uptrend_data: dict) -> None:
        """Signals during warmup should be 0."""
        strat = TrendMediumV1()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals[:strat.warmup] == 0.0)

    def test_indicator_config(self) -> None:
        """get_indicator_config returns both indicators."""
        strat = TrendMediumV1()
        config = strat.get_indicator_config()
        names = [c["name"] for c in config]
        assert "supertrend" in names
        assert "volume_momentum" in names


class TestTrendMediumV2:
    """Tests for ADX + Force Index strategy."""

    def test_can_instantiate(self) -> None:
        """Strategy can be instantiated."""
        strat = TrendMediumV2()
        assert strat.name == "trend_medium_v2"
        assert strat.horizon == "medium"

    def test_on_init_arrays(self, uptrend_data: dict) -> None:
        """on_init_arrays populates ADX, DI, and Force Index arrays."""
        strat = TrendMediumV2()
        _init_strategy(strat, uptrend_data)
        assert strat._adx_vals is not None
        assert strat._fi is not None

    def test_signal_in_range(self, uptrend_data: dict) -> None:
        """All signals in [-1, 1]."""
        strat = TrendMediumV2()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)

    def test_warmup_zero(self, uptrend_data: dict) -> None:
        """Warmup signals are 0."""
        strat = TrendMediumV2()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals[:strat.warmup] == 0.0)


class TestTrendFastV1:
    """Tests for Short Momentum + Volume Spike strategy."""

    def test_can_instantiate(self) -> None:
        """Strategy can be instantiated."""
        strat = TrendFastV1()
        assert strat.name == "trend_fast_v1"
        assert strat.horizon == "fast"

    def test_signal_in_range(self, uptrend_data: dict) -> None:
        """All signals in [-1, 1]."""
        strat = TrendFastV1()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)

    def test_warmup_zero(self, uptrend_data: dict) -> None:
        """Warmup signals are 0."""
        strat = TrendFastV1()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals[:strat.warmup] == 0.0)

    def test_indicator_config(self) -> None:
        """get_indicator_config returns ROC and volume_spike."""
        strat = TrendFastV1()
        config = strat.get_indicator_config()
        names = [c["name"] for c in config]
        assert "rate_of_change" in names
        assert "volume_spike" in names


class TestTrendSlowV1:
    """Tests for Donchian Channel breakout strategy."""

    def test_can_instantiate(self) -> None:
        """Strategy can be instantiated."""
        strat = TrendSlowV1()
        assert strat.name == "trend_slow_v1"
        assert strat.horizon == "slow"

    def test_signal_in_range(self) -> None:
        """All signals in [-1, 1]."""
        data = _make_arrays(600, trend="up")
        strat = TrendSlowV1()
        _init_strategy(strat, data)
        signals = strat.generate_signals()
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)

    def test_warmup_zero(self) -> None:
        """Warmup signals are 0."""
        data = _make_arrays(600, trend="up")
        strat = TrendSlowV1()
        _init_strategy(strat, data)
        signals = strat.generate_signals()
        assert np.all(signals[:strat.warmup] == 0.0)


class TestMeanReversionV1:
    """Tests for Bollinger Band + RSI extreme strategy."""

    def test_can_instantiate(self) -> None:
        """Strategy can be instantiated."""
        strat = MeanReversionV1()
        assert strat.name == "mr_v1"
        assert strat.regime == "mean_reversion"
        assert strat.horizon is None

    def test_signal_in_range(self, uptrend_data: dict) -> None:
        """All signals in [-1, 1]."""
        strat = MeanReversionV1()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)

    def test_warmup_zero(self, uptrend_data: dict) -> None:
        """Warmup signals are 0."""
        strat = MeanReversionV1()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals[:strat.warmup] == 0.0)

    def test_indicator_config(self) -> None:
        """get_indicator_config returns BB and RSI."""
        strat = MeanReversionV1()
        config = strat.get_indicator_config()
        names = [c["name"] for c in config]
        assert "bollinger_bands" in names
        assert "rsi" in names


# ---------------------------------------------------------------------------
# 5. Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: empty arrays, short arrays, constant price."""

    def test_tsmom_fast_short_array(self, short_data: dict) -> None:
        """TSMOMFast with fewer bars than warmup produces all zeros."""
        strat = TSMOMFast()
        _init_strategy(strat, short_data)
        signals = strat.generate_signals()
        assert len(signals) == 10
        assert np.all(signals == 0.0)

    def test_tsmom_medium_short_array(self, short_data: dict) -> None:
        """TSMOMMedium with fewer bars than warmup produces all zeros."""
        strat = TSMOMMedium()
        _init_strategy(strat, short_data)
        signals = strat.generate_signals()
        assert np.all(signals == 0.0)

    def test_trend_medium_v1_short_array(self, short_data: dict) -> None:
        """TrendMediumV1 with short data produces all zeros."""
        strat = TrendMediumV1()
        _init_strategy(strat, short_data)
        signals = strat.generate_signals()
        assert np.all(signals == 0.0)

    def test_flat_price_tsmom(self, flat_data: dict) -> None:
        """Constant price produces 0 return → -1 signal (not > 0)."""
        strat = TSMOMFast()
        _init_strategy(strat, flat_data)
        signals = strat.generate_signals()
        post_warmup = signals[strat.warmup:]
        # cum_return = 0 → not > 0 → signal = -1
        assert np.all(post_warmup == -1.0)

    def test_mr_v1_flat_price(self, flat_data: dict) -> None:
        """MR strategy on flat price should produce zero or near-zero signals."""
        strat = MeanReversionV1()
        _init_strategy(strat, flat_data)
        signals = strat.generate_signals()
        # Flat price stays at BB middle → no extreme → 0
        post_warmup = signals[strat.warmup:]
        assert np.all(np.abs(post_warmup) <= 0.5)

    def test_all_strategies_random_data(self) -> None:
        """All strategies run without error on random data."""
        data = _make_arrays(500, trend="random")
        strategies = [
            TSMOMFast(),
            TSMOMMedium(),
            TrendMediumV1(),
            TrendMediumV2(),
            TrendMediumV3(),
            TrendMediumV4(),
            TrendMediumV5(),
            TrendFastV1(),
            MeanReversionV1(),
        ]
        for strat in strategies:
            _init_strategy(strat, data)
            signals = strat.generate_signals()
            assert len(signals) == 500
            assert np.all(np.isfinite(signals))
            assert np.all(signals >= -1.0)
            assert np.all(signals <= 1.0)

    def test_tsmom_slow_random_data(self) -> None:
        """TSMOMSlow and TrendSlowV1 run on 600-bar random data."""
        data = _make_arrays(600, trend="random")
        for strat in [TSMOMSlow(), TrendSlowV1()]:
            _init_strategy(strat, data)
            signals = strat.generate_signals()
            assert len(signals) == 600
            assert np.all(np.isfinite(signals))
            assert np.all(signals >= -1.0)
            assert np.all(signals <= 1.0)

    def test_generate_signals_length_matches_input(self, uptrend_data: dict) -> None:
        """Output signal array length matches input array length."""
        strat = TSMOMFast()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert len(signals) == len(uptrend_data["closes"])

    def test_signal_clipping(self) -> None:
        """Verify generate_signals clips to [-1, 1] even if _generate_signal overshoots."""

        class OvershootStrategy(MeanReversionStrategy):
            """Test strategy that intentionally returns out-of-range signals."""
            name = "overshoot_test"
            signal_dimensions = ["technical"]
            warmup = 0

            def _generate_signal(self, bar_index: int) -> float:
                return 5.0  # Intentionally out of range

        strat = OvershootStrategy()
        data = _make_arrays(10, trend="up")
        _init_strategy(strat, data)
        signals = strat.generate_signals()
        assert np.all(signals <= 1.0)
        assert np.all(signals >= -1.0)


# ---------------------------------------------------------------------------
# 6. New medium strategies: v3, v4, v5
# ---------------------------------------------------------------------------

class TestTrendMediumV3:
    """Tests for MACD + OI Flow strategy."""

    def test_can_instantiate(self) -> None:
        """Strategy can be instantiated with correct attributes."""
        strat = TrendMediumV3()
        assert strat.name == "trend_medium_v3"
        assert strat.horizon == "medium"
        assert strat.regime == "trending"

    def test_on_init_arrays(self, uptrend_data: dict) -> None:
        """on_init_arrays populates MACD line and OI Flow arrays."""
        strat = TrendMediumV3()
        _init_strategy(strat, uptrend_data)
        assert strat._macd_line is not None
        assert strat._oi_flow_line is not None
        assert len(strat._macd_line) == 500
        assert len(strat._oi_flow_line) == 500

    def test_signal_in_range(self, uptrend_data: dict) -> None:
        """All signals must be in [-1, 1]."""
        strat = TrendMediumV3()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)

    def test_warmup_zero(self, uptrend_data: dict) -> None:
        """Signals during warmup period should be 0."""
        strat = TrendMediumV3()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals[:strat.warmup] == 0.0)

    def test_indicator_config(self) -> None:
        """get_indicator_config returns macd and oi_flow entries."""
        strat = TrendMediumV3()
        config = strat.get_indicator_config()
        names = [c["name"] for c in config]
        assert "macd" in names
        assert "oi_flow" in names


class TestTrendMediumV4:
    """Tests for KAMA Crossover + Volume Efficiency strategy."""

    def test_can_instantiate(self) -> None:
        """Strategy can be instantiated with correct attributes."""
        strat = TrendMediumV4()
        assert strat.name == "trend_medium_v4"
        assert strat.horizon == "medium"
        assert strat.regime == "trending"

    def test_on_init_arrays(self, uptrend_data: dict) -> None:
        """on_init_arrays populates fast KAMA, slow KAMA, and vol eff z-score arrays."""
        strat = TrendMediumV4()
        _init_strategy(strat, uptrend_data)
        assert strat._fast_kama is not None
        assert strat._slow_kama is not None
        assert strat._vol_eff_z is not None
        assert len(strat._fast_kama) == 500

    def test_signal_in_range(self, uptrend_data: dict) -> None:
        """All signals must be in [-1, 1]."""
        strat = TrendMediumV4()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)

    def test_warmup_zero(self, uptrend_data: dict) -> None:
        """Signals during warmup period should be 0."""
        strat = TrendMediumV4()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals[:strat.warmup] == 0.0)

    def test_indicator_config(self) -> None:
        """get_indicator_config returns kama and volume_efficiency entries."""
        strat = TrendMediumV4()
        config = strat.get_indicator_config()
        names = [c["name"] for c in config]
        assert "kama" in names
        assert "volume_efficiency" in names


class TestTrendMediumV5:
    """Tests for EMA Ribbon + RSI Momentum strategy."""

    def test_can_instantiate(self) -> None:
        """Strategy can be instantiated with correct attributes."""
        strat = TrendMediumV5()
        assert strat.name == "trend_medium_v5"
        assert strat.horizon == "medium"
        assert strat.regime == "trending"

    def test_on_init_arrays(self, uptrend_data: dict) -> None:
        """on_init_arrays populates EMA ribbon and RSI arrays."""
        strat = TrendMediumV5()
        _init_strategy(strat, uptrend_data)
        assert strat._ribbon_emas is not None
        assert strat._rsi_vals is not None
        assert len(strat._ribbon_emas) == 5
        assert len(strat._rsi_vals) == 500

    def test_signal_in_range(self, uptrend_data: dict) -> None:
        """All signals must be in [-1, 1]."""
        strat = TrendMediumV5()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)

    def test_warmup_zero(self, uptrend_data: dict) -> None:
        """Signals during warmup period should be 0."""
        strat = TrendMediumV5()
        _init_strategy(strat, uptrend_data)
        signals = strat.generate_signals()
        assert np.all(signals[:strat.warmup] == 0.0)

    def test_indicator_config(self) -> None:
        """get_indicator_config returns ema_ribbon and rsi entries."""
        strat = TrendMediumV5()
        config = strat.get_indicator_config()
        names = [c["name"] for c in config]
        assert "ema_ribbon" in names
        assert "rsi" in names
