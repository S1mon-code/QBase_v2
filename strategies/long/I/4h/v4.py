"""MildTrendLongI4hV4 — HMA(40) + ADX(30) + OI_Momentum(40).

Economic logic: HMA provides responsive trend detection with minimal lag for 4H iron
ore. ADX above 20 confirms the market is trending (not ranging). OI momentum shows
new positions being built. Signal strength scales with ADX directional strength and
OI expansion rate for gradual position sizing.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx
from indicators.trend.hma import hma
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV4(TrendingStrategy):
    name = "long_I_4h_v4"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 60

    hma_period: int = 40
    adx_period: int = 30
    oi_period: int = 40
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._adx = adx(self._highs, self._lows, self._closes, period=self.adx_period)
        self._oi_mom = oi_momentum(self._oi, period=self.oi_period)

    def _generate_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1] if bar_index > 0 else np.nan
        adx_val = self._adx[bar_index]
        oi_val = self._oi_mom[bar_index]

        if any(np.isnan(v) for v in [h, h_prev, adx_val, oi_val]):
            return 0.0

        hma_rising = h > h_prev
        adx_trending = adx_val > 20.0
        oi_expanding = oi_val > 0.0

        if not (hma_rising and adx_trending and oi_expanding):
            return 0.0

        adx_score = min(1.0, (adx_val - 20.0) / 30.0) * 0.4
        oi_score = min(1.0, oi_val / 0.10) * 0.3
        return min(1.0, 0.3 + adx_score + oi_score)

    def get_indicator_config(self):
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": f"ADX({self.adx_period})", "array": self._adx, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [20, 25]},
            {"name": f"OI Mom({self.oi_period})", "array": self._oi_mom, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "ADX",
                    [self._make_subplot_trace("ADX", datetimes, self._adx, color="#42a5f5")],
                    y_range=[0, 100], horizontal_lines=[20, 25],
                ),
                self._make_subplot(
                    "OI Momentum",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#4fc3f7")],
                    zero_line=True,
                ),
            ],
        }
