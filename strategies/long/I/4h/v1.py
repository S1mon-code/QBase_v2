"""MildTrendLongI4hV1 — EMA(30,60) + RSI(40) + OI_Momentum(50).

Economic logic: Dual EMA crossover captures multi-day iron ore trends on 4H bars.
RSI above 50 confirms bullish momentum without requiring overbought conditions.
OI expansion validates new speculative interest entering the trend. Signal scales
with RSI distance from neutral and OI growth rate.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.trend.ema import ema
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV1(TrendingStrategy):
    name = "long_I_4h_v1"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 70

    ema_fast: int = 30
    ema_slow: int = 60
    rsi_period: int = 40
    oi_period: int = 50
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = ema(self._closes, period=self.ema_fast)
        self._ema_slow = ema(self._closes, period=self.ema_slow)
        self._rsi = rsi(self._closes, period=self.rsi_period)
        self._oi_mom = oi_momentum(self._oi, period=self.oi_period)

    def _generate_signal(self, bar_index: int) -> float:
        ef = self._ema_fast[bar_index]
        es = self._ema_slow[bar_index]
        r = self._rsi[bar_index]
        oi_val = self._oi_mom[bar_index]

        if any(np.isnan(v) for v in [ef, es, r, oi_val]):
            return 0.0

        if not (ef > es and r > 50.0 and oi_val > 0.0):
            return 0.0

        rsi_score = min(1.0, (r - 50.0) / 30.0) * 0.4
        oi_score = min(1.0, oi_val / 0.10) * 0.3
        return min(1.0, 0.3 + rsi_score + oi_score)

    def get_indicator_config(self):
        return [
            {"name": f"EMA({self.ema_fast})", "array": self._ema_fast, "type": "overlay"},
            {"name": f"EMA({self.ema_slow})", "array": self._ema_slow, "type": "overlay"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [30, 70]},
            {"name": f"OI Mom({self.oi_period})", "array": self._oi_mom, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_fast})", datetimes, self._ema_fast, color="#ffab40"),
                self._make_overlay(f"EMA({self.ema_slow})", datetimes, self._ema_slow, color="#ab47bc"),
            ],
            "subplots": [
                self._make_subplot(
                    "RSI",
                    [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#bb86fc")],
                    y_range=[0, 100], horizontal_lines=[30, 50, 70],
                ),
                self._make_subplot(
                    "OI Momentum",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#4fc3f7")],
                    zero_line=True,
                ),
            ],
        }
