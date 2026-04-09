"""MildTrendLongI2hV4 — HMA(50) + TSI(40,20) + CMF(60).

Economic logic: HMA provides responsive low-lag trend tracking for 2H iron ore.
TSI double-smoothed momentum filters noise while confirming trend direction.
CMF validates institutional money flow. Signal scales with TSI magnitude and CMF
level for gradual position sizing.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.trend.hma import hma
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV4(TrendingStrategy):
    name = "mild_trend_long_I_2h_v4"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 70

    hma_period: int = 50
    tsi_long: int = 40
    tsi_short: int = 20
    cmf_period: int = 60
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=self.tsi_short
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, period=self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1] if bar_index > 0 else np.nan
        tsi_val = self._tsi_line[bar_index]
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in [h, h_prev, tsi_val, cmf_val]):
            return 0.0

        hma_rising = h > h_prev
        tsi_bullish = tsi_val > 0.0
        cmf_positive = cmf_val > 0.0

        if not (hma_rising and tsi_bullish and cmf_positive):
            return 0.0

        tsi_score = min(1.0, tsi_val / 20.0) * 0.4
        cmf_score = min(1.0, cmf_val / 0.25) * 0.3
        return min(1.0, 0.3 + tsi_score + cmf_score)

    def get_indicator_config(self):
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": "TSI", "array": self._tsi_line, "type": "subplot", "panel": "TSI", "zero_line": True},
            {"name": "TSI Signal", "array": self._tsi_signal, "type": "subplot", "panel": "TSI", "style": "dash"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "TSI",
                    [
                        self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#bb86fc"),
                        self._make_subplot_trace("Signal", datetimes, self._tsi_signal, color="#ff8a80", style="dash"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "CMF",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
