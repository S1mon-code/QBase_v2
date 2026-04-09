"""StrongTrendLongAG2hV5 — LinearReg(80) + Vortex(50) + VolumeSpike(35).

Economic logic: Linear Regression on 2H bars captures AG's directional slope.
Vortex separates positive and negative trend movement. Volume spike detection
confirms breakout conviction on Silver's volatile 2H moves.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression
from indicators.trend.vortex import vortex
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG2hV5(TrendingStrategy):
    name = "long_AG_2h_v5"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 100

    linreg_period: int = 80
    vortex_period: int = 50
    vol_spike_period: int = 35
    vol_spike_thresh: float = 2.0
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._linreg = linear_regression(self._closes, period=self.linreg_period)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period,
        )
        self._vol_spike = volume_spike(
            self._volumes, period=self.vol_spike_period, threshold=self.vol_spike_thresh,
        )

    def _generate_signal(self, bar_index: int) -> float:
        lr = self._linreg[bar_index]
        lr_prev = self._linreg[bar_index - 1]
        vip = self._vi_plus[bar_index]
        vim = self._vi_minus[bar_index]

        if any(np.isnan(v) for v in (lr, lr_prev, vip, vim)):
            return 0.0

        lr_rising = lr > lr_prev
        vortex_bull = vip > vim
        spike = bool(self._vol_spike[bar_index]) if bar_index < len(self._vol_spike) else False

        if lr_rising and vortex_bull:
            diff = vip - vim
            base = min(1.0, diff * 2.5 + 0.3)
            if spike:
                base = min(1.0, base + 0.2)
            return max(0.0, base)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"LinReg({self.linreg_period})", "array": self._linreg, "type": "overlay"},
            {"name": f"VI+({self.vortex_period})", "array": self._vi_plus,
             "type": "subplot", "panel": "Vortex"},
            {"name": f"VI-({self.vortex_period})", "array": self._vi_minus,
             "type": "subplot", "panel": "Vortex", "horizontal_lines": [1.0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"LinReg({self.linreg_period})", datetimes, self._linreg, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Vortex({self.vortex_period})",
                    [self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#66bb6a"),
                     self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
