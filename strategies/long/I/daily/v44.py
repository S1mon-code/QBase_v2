"""MildTrendLongIDailyV44 — KAMA(100) slope + Williams %R(20) > -40 + Volume spike(20).

Economic logic: KAMA slope captures adaptive trend direction while filtering noise.
Williams %R above -40 shows bullish momentum without extreme overbought. Volume spike
confirms institutional participation, validating the price move.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.momentum.williams_r import williams_r
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV44(TrendingStrategy):
    name = "long_I_daily_v44"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 130

    kama_period: int = 100
    wr_period: int = 20
    vol_spike_period: int = 20
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._wr = williams_r(self._highs, self._lows, self._closes, period=self.wr_period)
        self._vol_spike = volume_spike(self._volumes, period=self.vol_spike_period)

    def _generate_signal(self, bar_index: int) -> float:
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1] if bar_index > 0 else np.nan
        wr_val = self._wr[bar_index]
        vs = self._vol_spike[bar_index]

        if any(np.isnan(v) for v in [k, k_prev, wr_val, vs]):
            return 0.0

        kama_rising = k > k_prev
        wr_bullish = wr_val > -40.0
        vol_active = vs > 1.0

        if not (kama_rising and wr_bullish and vol_active):
            return 0.0

        # Scale by KAMA slope magnitude
        slope = (k - k_prev) / k_prev * 1000.0 if k_prev != 0 else 0.0
        slope_score = min(0.5, max(0.0, slope / 5.0)) + 0.3
        # Boost with volume spike
        vol_boost = min(0.2, (vs - 1.0) * 0.1)
        return min(1.0, slope_score + vol_boost)

    def get_indicator_config(self):
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": f"Williams %R({self.wr_period})", "array": self._wr, "type": "subplot",
             "y_range": [-100, 0], "horizontal_lines": [-40]},
            {"name": f"Vol Spike({self.vol_spike_period})", "array": self._vol_spike,
             "type": "subplot", "horizontal_lines": [1.0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama,
                                   color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Williams %R({self.wr_period})",
                    [self._make_subplot_trace("W%R", datetimes, self._wr, color="#ab47bc")],
                    horizontal_lines=[-40], y_range=[-100, 0],
                ),
                self._make_subplot(
                    "Volume Spike",
                    [self._make_subplot_trace("Spike", datetimes, self._vol_spike,
                                             color="#ff7043")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
