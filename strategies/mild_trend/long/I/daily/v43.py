"""MildTrendLongIDailyV43 — Ichimoku cloud breakout + RSI(18) > 55.

Economic logic: Ichimoku cloud provides a comprehensive view of support/resistance
and trend direction. Price breaking above the cloud signals bullish regime shift.
RSI above 55 confirms momentum without being overbought. Signal scales with distance
above cloud and RSI strength.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ichimoku import ichimoku
from indicators.momentum.rsi import rsi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV43(TrendingStrategy):
    name = "mild_trend_long_I_daily_v43"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup = 80

    ichi_tenkan: int = 9
    ichi_kijun: int = 26
    rsi_period: int = 18
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tenkan, self._kijun, self._senkou_a, self._senkou_b, _ = ichimoku(
            self._highs, self._lows, self._closes,
            tenkan=self.ichi_tenkan, kijun=self.ichi_kijun)
        self._rsi = rsi(self._closes, period=self.rsi_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        sa = self._senkou_a[bar_index]
        sb = self._senkou_b[bar_index]
        rsi_val = self._rsi[bar_index]

        if any(np.isnan(v) for v in [close, sa, sb, rsi_val]):
            return 0.0

        cloud_top = max(sa, sb)
        above_cloud = close > cloud_top
        rsi_ok = rsi_val > 55.0

        if not (above_cloud and rsi_ok):
            return 0.0

        # Scale by distance above cloud
        cloud_dist = (close - cloud_top) / close * 100.0
        dist_score = min(0.5, max(0.0, cloud_dist / 5.0)) + 0.3
        # Boost with RSI
        rsi_boost = min(0.2, (rsi_val - 55.0) / 100.0)
        return min(1.0, dist_score + rsi_boost)

    def get_indicator_config(self):
        return [
            {"name": "Senkou A", "array": self._senkou_a, "type": "overlay"},
            {"name": "Senkou B", "array": self._senkou_b, "type": "overlay"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [55]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("Senkou A", datetimes, self._senkou_a, color="#26a69a"),
                self._make_overlay("Senkou B", datetimes, self._senkou_b, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    f"RSI({self.rsi_period})",
                    [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#42a5f5")],
                    horizontal_lines=[55], y_range=[0, 100],
                ),
            ],
        }
