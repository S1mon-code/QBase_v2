"""MACD Histogram Momentum + Volume Surge — Accelerating momentum confirmed by volume expansion.

The MACD histogram captures the rate of change of the MACD line relative to its
signal line.  When the histogram is rising (accelerating momentum) and volume
exceeds its rolling mean, it signals that fresh capital is driving the move.
This combination filters for high-conviction iron ore breakouts on the 1H chart.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV19(TrendingStrategy):
    """Long signal on accelerating MACD histogram with volume surge.

    Signal logic
    ------------
    * macd_hist > 0 AND rising AND vol_ratio > 1.2
        ->  min(1.0, 0.4 + vol_ratio / 3.0)
    * macd_hist > 0 AND macd_line > 0
        ->  0.3
    * else  ->  0.0
    """

    name = "mild_trend_long_I_1h_v19"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    fast_period: int = 20
    slow_period: int = 50
    signal_period: int = 12
    vol_period: int = 30
    chandelier_mult: float = 2.5

    def on_init_arrays(
        self, closes, highs, lows, opens, volumes, oi, datetimes
    ):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._macd_line, self._signal_line, self._histogram = macd(
            closes, self.fast_period, self.slow_period, self.signal_period
        )
        kernel = np.ones(self.vol_period) / self.vol_period
        self._vol_mean = np.convolve(volumes, kernel, mode="same")

    def _generate_signal(self, bar_index: int) -> float:
        hist = self._histogram[bar_index]
        macd_line = self._macd_line[bar_index]
        vol = self._volumes[bar_index]
        vol_mean = self._vol_mean[bar_index]

        if bar_index < 1:
            return 0.0

        prev_hist = self._histogram[bar_index - 1]
        vol_ratio = vol / vol_mean if vol_mean > 0 else 0.0

        if hist > 0 and hist > prev_hist and vol_ratio > 1.2:
            return min(1.0, 0.4 + vol_ratio / 3.0)
        if hist > 0 and macd_line > 0:
            return 0.3
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "MACD Line", "array": self._macd_line, "panel": "MACD"},
            {"name": "Signal Line", "array": self._signal_line, "panel": "MACD"},
            {"name": "Histogram", "array": self._histogram, "panel": "MACD", "style": "bar"},
            {"name": "Vol Mean", "array": self._vol_mean},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"MACD({self.fast_period},{self.slow_period},{self.signal_period})",
                [
                    self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._signal_line, color="#4fc3f7"),
                    self._make_subplot_trace("Histogram", datetimes, self._histogram, style="bar", color_positive="#26a69a", color_negative="#ef5350"),
                ],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
