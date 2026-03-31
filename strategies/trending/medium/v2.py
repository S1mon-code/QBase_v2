"""Trend Medium V2 — ADX + Force Index.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: ADX measures trend strength regardless of direction.  When
ADX is above threshold the market is trending; Force Index (Elder) combines
price change with volume to reveal the directional bias of large players.

Who loses money: Counter-trend traders who fade strong trends — the Force
Index exposes whether institutional capital is driving the move.
"""

from __future__ import annotations

import numpy as np

from strategies.templates.trending_template import TrendingStrategy
from indicators.trend.adx import adx_with_di
from indicators.volume.force_index import force_index


class TrendMediumV2(TrendingStrategy):
    """ADX trend filter + Force Index directional signal.

    Full signal when ADX > threshold and Force Index confirms direction.
    Zero signal when ADX indicates no trend.

    Attributes:
        adx_period:      ADX lookback period.
        adx_threshold:   Minimum ADX to consider market trending.
        fi_period:       Force Index EMA period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "trend_medium_v2"
    horizon = "medium"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 150

    adx_period: int = 14
    adx_threshold: float = 25.0
    fi_period: int = 13
    chandelier_mult: float = 2.5

    # --- Precomputed arrays ---
    _adx_vals: np.ndarray | None = None
    _plus_di: np.ndarray | None = None
    _minus_di: np.ndarray | None = None
    _fi: np.ndarray | None = None

    def on_init_arrays(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        volumes: np.ndarray,
        oi: np.ndarray,
        datetimes: np.ndarray,
    ) -> None:
        """Precompute ADX, DI, and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._adx_vals, self._plus_di, self._minus_di = adx_with_di(
            self._highs, self._lows, self._closes, period=self.adx_period,
        )
        self._fi = force_index(
            self._closes, self._volumes, period=self.fi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Combine ADX filter with Force Index direction."""
        adx_val = self._adx_vals[bar_index]
        fi_val = self._fi[bar_index]
        plus = self._plus_di[bar_index]
        minus = self._minus_di[bar_index]

        if np.isnan(adx_val) or np.isnan(fi_val):
            return 0.0

        # No signal when trend is weak
        if adx_val < self.adx_threshold:
            return 0.0

        # Direction from DI crossover, confirmed by Force Index sign
        di_direction = 1.0 if plus > minus else -1.0
        fi_direction = 1.0 if fi_val > 0 else -1.0

        if di_direction == fi_direction:
            return di_direction
        # Conflicting signals → half conviction from DI
        return di_direction * 0.5

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "adx", "params": {"period": self.adx_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]
