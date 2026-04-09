"""RSI Momentum + EMA Trend Filter — Trend-following via RSI momentum with EMA directional filter.

Iron ore tends to trend strongly on hourly timeframes when momentum aligns with
the broader trend direction.  RSI measures the speed of price changes while the
EMA acts as a regime filter, ensuring signals fire only when the market is
already trading above its moving average.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV16(TrendingStrategy):
    """Long signal when RSI shows momentum above EMA trend filter.

    Signal logic
    ------------
    * close > ema AND rsi > 55  ->  min(1.0, (rsi - 50) / 30)
    * close > ema AND rsi > 40  ->  0.3  (weak bullish)
    * else                      ->  0.0
    """

    name = "long_I_1h_v16"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 50

    rsi_period: int = 20
    ema_period: int = 40
    chandelier_mult: float = 2.5

    def on_init_arrays(
        self, closes, highs, lows, opens, volumes, oi, datetimes
    ):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._rsi = rsi(closes, self.rsi_period)
        self._ema = ema(closes, self.ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        rsi_val = self._rsi[bar_index]
        ema_val = self._ema[bar_index]

        if close > ema_val and rsi_val > 55:
            return min(1.0, (rsi_val - 50) / 30)
        if close > ema_val and rsi_val > 40:
            return 0.3
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"EMA({self.ema_period})", "array": self._ema},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"EMA({self.ema_period})", datetimes, self._ema, color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"RSI({self.rsi_period})",
                [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc")],
                horizontal_lines=[30, 70], y_range=[0, 100],
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
