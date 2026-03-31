"""Trend Medium V5 — EMA Ribbon + RSI Momentum.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: EMA Ribbon alignment (all EMAs stacked in order:
8 < 13 < 21 < 34 < 55 for uptrend) indicates a strong, structured trend
across multiple timeframes.  RSI confirms momentum direction and filters
extremes — in a healthy trend, RSI should stay in the 40-70 range (not
overbought).

Who loses money: Mean-reverters who fight ribbon-aligned trends.
"""

from __future__ import annotations

import numpy as np

from strategies.templates.trending_template import TrendingStrategy
from indicators.trend.ema_ribbon import ema_ribbon
from indicators.momentum.rsi import rsi


class TrendMediumV5(TrendingStrategy):
    """EMA Ribbon full alignment + RSI momentum filter.

    Full signal when all 5 EMAs are stacked in order and RSI confirms momentum
    direction.  Half signal when ribbon is aligned but RSI filter fails (trend
    present but momentum fading).  Zero signal when ribbon is not aligned.

    Attributes:
        ribbon_periods:  Tuple of EMA periods for the ribbon (class variable).
        rsi_period:      RSI lookback period.
        rsi_low:         RSI threshold — above this for longs, below (100 - rsi_low) for shorts.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "trend_medium_v5"
    horizon = "medium"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 120

    # ribbon_periods is a tuple — not type-annotated as int/float (not optimisable)
    ribbon_periods = (8, 13, 21, 34, 55)

    rsi_period: int = 14
    rsi_low: float = 40.0
    chandelier_mult: float = 2.5

    # --- Precomputed arrays ---
    _ribbon_emas: list | None = None
    _rsi_vals: np.ndarray | None = None

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
        """Precompute EMA Ribbon and RSI arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._ribbon_emas = ema_ribbon(self._closes, periods=self.ribbon_periods)
        self._rsi_vals = rsi(self._closes, period=self.rsi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Combine EMA Ribbon alignment with RSI momentum filter."""
        i = bar_index

        # Check for NaN in any ribbon EMA
        for ema_arr in self._ribbon_emas:
            if np.isnan(ema_arr[i]):
                return 0.0

        rsi_val = self._rsi_vals[i]
        if np.isnan(rsi_val):
            return 0.0

        bull_aligned = (
            self._ribbon_emas[0][i] > self._ribbon_emas[1][i]
            and self._ribbon_emas[1][i] > self._ribbon_emas[2][i]
            and self._ribbon_emas[2][i] > self._ribbon_emas[3][i]
            and self._ribbon_emas[3][i] > self._ribbon_emas[4][i]
        )
        bear_aligned = (
            self._ribbon_emas[0][i] < self._ribbon_emas[1][i]
            and self._ribbon_emas[1][i] < self._ribbon_emas[2][i]
            and self._ribbon_emas[2][i] < self._ribbon_emas[3][i]
            and self._ribbon_emas[3][i] < self._ribbon_emas[4][i]
        )

        if bull_aligned:
            rsi_long_ok = rsi_val > self.rsi_low
            return 1.0 if rsi_long_ok else 0.5

        if bear_aligned:
            rsi_short_ok = rsi_val < (100.0 - self.rsi_low)
            return -1.0 if rsi_short_ok else -0.5

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "ema_ribbon",
                "params": {"periods": self.ribbon_periods},
            },
            {
                "name": "rsi",
                "params": {"period": self.rsi_period},
            },
        ]
