"""MildTrendLongIDailyV3 — LinearReg(200) + TSI(60,30) + OI_Momentum(80).

Economic logic: Linear regression line acts as a slow trend anchor for iron ore's
structural moves. TSI double-smooths momentum to filter daily noise while maintaining
responsiveness. OI expansion confirms new money entering the market in the trend
direction. Three independent confirmation dimensions reduce false signals.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.trend.linear_regression import linear_regression
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV3(TrendingStrategy):
    name = "long_I_daily_v3"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 220

    lr_period: int = 200
    tsi_long: int = 60
    tsi_short: int = 30
    oi_period: int = 80
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._lr = linear_regression(self._closes, period=self.lr_period)
        self._tsi_line, self._tsi_signal = tsi(self._closes, long_period=self.tsi_long, short_period=self.tsi_short)
        self._oi_mom = oi_momentum(self._oi, period=self.oi_period)

    def _generate_signal(self, bar_index: int) -> float:
        lr_val = self._lr[bar_index]
        tsi_val = self._tsi_line[bar_index]
        oi_val = self._oi_mom[bar_index]
        close = self._closes[bar_index]

        if any(np.isnan(v) for v in [lr_val, tsi_val, oi_val, close]):
            return 0.0

        above_lr = close > lr_val
        tsi_bullish = tsi_val > 0.0
        oi_expanding = oi_val > 0.0

        if not (above_lr and tsi_bullish and oi_expanding):
            return 0.0

        # Scale with TSI magnitude (0-25 range) and OI expansion
        tsi_score = min(1.0, tsi_val / 25.0) * 0.5
        oi_score = min(1.0, oi_val / 0.15) * 0.3
        return min(1.0, 0.2 + tsi_score + oi_score)

    def get_indicator_config(self):
        return [
            {"name": f"LinReg({self.lr_period})", "array": self._lr, "type": "overlay"},
            {"name": f"TSI({self.tsi_long},{self.tsi_short})", "array": self._tsi_line, "type": "subplot", "zero_line": True},
            {"name": f"OI Mom({self.oi_period})", "array": self._oi_mom, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"LinReg({self.lr_period})", datetimes, self._lr, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "TSI",
                    [self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#bb86fc"),
                     self._make_subplot_trace("Signal", datetimes, self._tsi_signal, color="#ff8a80", style="dash")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "OI Momentum",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#4fc3f7")],
                    zero_line=True,
                ),
            ],
        }
