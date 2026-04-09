"""MildTrendLongI1hV2 — HMA(25) + MACD(10,25,8) + MFI(30).

Economic logic: HMA provides responsive trend detection for 1H iron ore with minimal
lag. MACD histogram confirms momentum acceleration. MFI above 50 validates
volume-weighted buying pressure. Signal scales with MACD histogram magnitude and
MFI strength for intraday trend capture.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.hma import hma
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV2(TrendingStrategy):
    name = "long_I_1h_v2"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 40

    hma_period: int = 25
    macd_fast: int = 10
    macd_slow: int = 25
    mfi_period: int = 30
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow, signal=8
        )
        self._mfi = mfi(self._highs, self._lows, self._closes, self._volumes, period=self.mfi_period)

    def _generate_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1] if bar_index > 0 else np.nan
        mh = self._macd_hist[bar_index]
        mfi_val = self._mfi[bar_index]

        if any(np.isnan(v) for v in [h, h_prev, mh, mfi_val]):
            return 0.0

        hma_rising = h > h_prev
        macd_bullish = mh > 0.0
        mfi_above_50 = mfi_val > 50.0

        if not (hma_rising and macd_bullish and mfi_above_50):
            return 0.0

        close = self._closes[bar_index]
        macd_score = min(1.0, mh / (close * 0.005)) * 0.4 if close > 0 else 0.0
        mfi_score = min(1.0, (mfi_val - 50.0) / 30.0) * 0.3
        return min(1.0, 0.3 + macd_score + mfi_score)

    def get_indicator_config(self):
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": "MACD Hist", "array": self._macd_hist, "type": "subplot", "panel": "MACD",
             "style": "bar", "color_positive": "#66bb6a", "color_negative": "#ef5350", "zero_line": True},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40"),
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
