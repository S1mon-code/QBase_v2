"""MildTrendLongIDailyV8 — HMA(100) + Vortex(80) + ForceIndex(60).

Economic logic: HMA's weighted-moving-average construction provides responsive trend
detection with minimal lag on daily iron ore. Vortex indicator measures positive vs
negative trend movement — VI+ > VI- confirms bullish pressure. Force Index combines
price change with volume for conviction measure. Signal scales with vortex differential.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.trend.vortex import vortex
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV8(TrendingStrategy):
    name = "mild_trend_long_I_daily_v8"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 120

    hma_period: int = 100
    vortex_period: int = 80
    fi_period: int = 60
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period
        )
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1] if bar_index > 0 else np.nan
        vip = self._vi_plus[bar_index]
        vim = self._vi_minus[bar_index]
        fi = self._fi[bar_index]

        if any(np.isnan(v) for v in [h, h_prev, vip, vim, fi]):
            return 0.0

        hma_rising = h > h_prev
        vortex_bullish = vip > vim
        fi_positive = fi > 0.0

        if not (hma_rising and vortex_bullish and fi_positive):
            return 0.0

        # Scale with vortex differential
        vortex_diff = vip - vim
        vortex_score = min(1.0, vortex_diff / 0.3) * 0.5
        return min(1.0, 0.3 + vortex_score + 0.2)

    def get_indicator_config(self):
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": "VI+", "array": self._vi_plus, "type": "subplot", "panel": "Vortex"},
            {"name": "VI-", "array": self._vi_minus, "type": "subplot", "panel": "Vortex"},
            {"name": f"Force Index({self.fi_period})", "array": self._fi, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Vortex",
                    [
                        self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#66bb6a"),
                        self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
                    ],
                    horizontal_lines=[1.0],
                ),
                self._make_subplot(
                    "Force Index",
                    [self._make_subplot_trace("FI", datetimes, self._fi, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
