"""MildTrendLongI1hV1 — EMA(20,45) + RSI(25) + VolumeMomentum(30).

Economic logic: Dual EMA crossover captures intraday iron ore trends on 1H bars.
RSI above 50 confirms bullish momentum. Volume momentum ratio above 1.0 validates
above-average participation. Signal scales with RSI distance from neutral and
volume conviction for gradual sizing.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.trend.ema import ema
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV1(TrendingStrategy):
    name = "long_I_1h_v1"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 55

    ema_fast: int = 20
    ema_slow: int = 45
    rsi_period: int = 25
    vol_mom_period: int = 30
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = ema(self._closes, period=self.ema_fast)
        self._ema_slow = ema(self._closes, period=self.ema_slow)
        self._rsi = rsi(self._closes, period=self.rsi_period)
        self._vol_mom = volume_momentum(self._volumes, period=self.vol_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        ef = self._ema_fast[bar_index]
        es = self._ema_slow[bar_index]
        r = self._rsi[bar_index]
        vm = self._vol_mom[bar_index]

        if any(np.isnan(v) for v in [ef, es, r, vm]):
            return 0.0

        if not (ef > es and r > 50.0 and vm > 1.0):
            return 0.0

        rsi_score = min(1.0, (r - 50.0) / 30.0) * 0.4
        vol_score = min(1.0, (vm - 1.0) / 0.5) * 0.3
        return min(1.0, 0.3 + rsi_score + vol_score)

    def get_indicator_config(self):
        return [
            {"name": f"EMA({self.ema_fast})", "array": self._ema_fast, "type": "overlay"},
            {"name": f"EMA({self.ema_slow})", "array": self._ema_slow, "type": "overlay"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [30, 70]},
            {"name": "Vol Mom", "array": self._vol_mom, "type": "subplot", "horizontal_lines": [1.0]},
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
                    "Volume Momentum",
                    [self._make_subplot_trace("VM", datetimes, self._vol_mom, color="#66bb6a")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
