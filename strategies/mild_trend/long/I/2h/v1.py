"""MildTrendLongI2hV1 — EMA(40,80) + MACD(15,40,10) + MFI(50).

Economic logic: Dual EMA crossover captures multi-session iron ore trends on 2H.
MACD with moderate parameters detects momentum shifts. MFI above 50 validates
volume-weighted buying pressure. Signal scales with MACD histogram magnitude and
MFI strength for gradual position entry.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.ema import ema
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV1(TrendingStrategy):
    name = "mild_trend_long_I_2h_v1"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 90

    ema_fast: int = 40
    ema_slow: int = 80
    macd_fast: int = 15
    macd_slow: int = 40
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = ema(self._closes, period=self.ema_fast)
        self._ema_slow = ema(self._closes, period=self.ema_slow)
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow, signal=10
        )
        self._mfi = mfi(self._highs, self._lows, self._closes, self._volumes, period=50)

    def _generate_signal(self, bar_index: int) -> float:
        ef = self._ema_fast[bar_index]
        es = self._ema_slow[bar_index]
        mh = self._macd_hist[bar_index]
        mfi_val = self._mfi[bar_index]

        if any(np.isnan(v) for v in [ef, es, mh, mfi_val]):
            return 0.0

        if not (ef > es and mh > 0.0 and mfi_val > 50.0):
            return 0.0

        close = self._closes[bar_index]
        macd_score = min(1.0, mh / (close * 0.008)) * 0.4 if close > 0 else 0.0
        mfi_score = min(1.0, (mfi_val - 50.0) / 30.0) * 0.3
        return min(1.0, 0.3 + macd_score + mfi_score)

    def get_indicator_config(self):
        return [
            {"name": f"EMA({self.ema_fast})", "array": self._ema_fast, "type": "overlay"},
            {"name": f"EMA({self.ema_slow})", "array": self._ema_slow, "type": "overlay"},
            {"name": "MACD Hist", "array": self._macd_hist, "type": "subplot", "panel": "MACD",
             "style": "bar", "color_positive": "#66bb6a", "color_negative": "#ef5350", "zero_line": True},
            {"name": "MFI", "array": self._mfi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_fast})", datetimes, self._ema_fast, color="#ffab40"),
                self._make_overlay(f"EMA({self.ema_slow})", datetimes, self._ema_slow, color="#ab47bc"),
            ],
            "subplots": [
                self._make_subplot(
                    "MACD",
                    [
                        self._make_subplot_trace("Line", datetimes, self._macd_line, color="#42a5f5"),
                        self._make_subplot_trace("Signal", datetimes, self._macd_signal, color="#ff8a80", style="dash"),
                        self._make_subplot_trace("Hist", datetimes, self._macd_hist, style="bar",
                                                 color_positive="#66bb6a", color_negative="#ef5350"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "MFI",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#26a69a")],
                    y_range=[0, 100], horizontal_lines=[20, 50, 80],
                ),
            ],
        }
