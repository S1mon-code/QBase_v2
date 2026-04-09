"""MildTrendLongIDailyV4 — FRAMA(120) + Aroon(120) + MFI(80).

Economic logic: FRAMA's fractal-adaptive smoothing naturally adjusts to iron ore's
shifting volatility regimes. Aroon identifies trend maturity — Aroon Up near 100
confirms a strong uptrend. MFI (volume-weighted RSI) validates institutional
participation. Signal blends price position vs FRAMA, Aroon strength, and MFI level.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.trend.aroon import aroon
from indicators.trend.frama import frama
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV4(TrendingStrategy):
    name = "mild_trend_long_I_daily_v4"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 140

    frama_period: int = 120
    aroon_period: int = 120
    mfi_period: int = 80
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._frama = frama(self._closes, period=self.frama_period)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, period=self.aroon_period
        )
        self._mfi = mfi(self._highs, self._lows, self._closes, self._volumes, period=self.mfi_period)

    def _generate_signal(self, bar_index: int) -> float:
        fr = self._frama[bar_index]
        ar_up = self._aroon_up[bar_index]
        ar_down = self._aroon_down[bar_index]
        mfi_val = self._mfi[bar_index]
        close = self._closes[bar_index]

        if any(np.isnan(v) for v in [fr, ar_up, ar_down, mfi_val, close]):
            return 0.0

        above_frama = close > fr
        aroon_bullish = ar_up > ar_down and ar_up > 50.0
        mfi_confirms = mfi_val > 40.0

        if not (above_frama and aroon_bullish and mfi_confirms):
            return 0.0

        aroon_score = min(1.0, (ar_up - 50.0) / 50.0) * 0.4
        mfi_score = min(1.0, (mfi_val - 40.0) / 40.0) * 0.3
        return min(1.0, 0.3 + aroon_score + mfi_score)

    def get_indicator_config(self):
        return [
            {"name": f"FRAMA({self.frama_period})", "array": self._frama, "type": "overlay"},
            {"name": f"Aroon Up({self.aroon_period})", "array": self._aroon_up, "type": "subplot", "panel": "Aroon"},
            {"name": f"Aroon Down({self.aroon_period})", "array": self._aroon_down, "type": "subplot", "panel": "Aroon"},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"FRAMA({self.frama_period})", datetimes, self._frama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Aroon",
                    [
                        self._make_subplot_trace("Up", datetimes, self._aroon_up, color="#66bb6a"),
                        self._make_subplot_trace("Down", datetimes, self._aroon_down, color="#ef5350"),
                    ],
                    y_range=[0, 100], horizontal_lines=[50],
                ),
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#42a5f5")],
                    y_range=[0, 100], horizontal_lines=[20, 50, 80],
                ),
            ],
        }
