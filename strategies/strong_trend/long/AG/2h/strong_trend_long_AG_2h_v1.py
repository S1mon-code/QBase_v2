"""StrongTrendLongAG2hV1 — EMA(35,75) + MACD(14,38,10) + RSI(50).

Economic logic: Dual EMA crossover captures Silver's 2H trend transitions.
MACD with moderate parameters tracks intermediate momentum. RSI with wide
period confirms sustained buying pressure above neutral.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.momentum.macd import macd
from indicators.momentum.rsi import rsi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG2hV1(TrendingStrategy):
    name = "strong_trend_long_AG_2h_v1"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 100

    ema_fast: int = 35
    ema_slow: int = 75
    macd_fast: int = 14
    macd_slow: int = 38
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = ema(self._closes, period=self.ema_fast)
        self._ema_slow = ema(self._closes, period=self.ema_slow)
        self._macd_line, self._macd_signal, _ = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow, signal=10,
        )
        self._rsi = rsi(self._closes, period=50)

    def _generate_signal(self, bar_index: int) -> float:
        ef = self._ema_fast[bar_index]
        es = self._ema_slow[bar_index]
        ml = self._macd_line[bar_index]
        r = self._rsi[bar_index]

        if any(np.isnan(v) for v in (ef, es, ml, r)):
            return 0.0

        if ef > es and ml > 0.0 and r > 50.0:
            strength = min(1.0, (r - 50.0) / 30.0 * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"EMA({self.ema_fast})", "array": self._ema_fast, "type": "overlay"},
            {"name": f"EMA({self.ema_slow})", "array": self._ema_slow, "type": "overlay"},
            {"name": f"MACD({self.macd_fast},{self.macd_slow})", "array": self._macd_line,
             "type": "subplot", "zero_line": True},
            {"name": "RSI(50)", "array": self._rsi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [30, 50, 70]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_fast})", datetimes, self._ema_fast, color="#ffab40"),
                self._make_overlay(f"EMA({self.ema_slow})", datetimes, self._ema_slow, color="#ab47bc"),
            ],
            "subplots": [
                self._make_subplot(
                    f"MACD({self.macd_fast},{self.macd_slow})",
                    [self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#42a5f5"),
                     self._make_subplot_trace("Signal", datetimes, self._macd_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "RSI(50)",
                    [self._make_subplot_trace("RSI", datetimes, self._rsi, color="#66bb6a")],
                    y_range=[0, 100], horizontal_lines=[30, 50, 70],
                ),
            ],
        }
