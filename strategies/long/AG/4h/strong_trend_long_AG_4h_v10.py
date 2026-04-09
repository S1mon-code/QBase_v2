"""StrongTrendLongAG4hV10 — FRAMA(50) + Williams%R(35) + MFI(40).

Economic logic: FRAMA adapts to AG's 4H fractal dimension for responsive
trend detection. Williams %R captures overbought/oversold momentum states
with wide lookback. MFI provides volume-weighted oscillator confirmation.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.frama import frama
from indicators.momentum.williams_r import williams_r
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG4hV10(TrendingStrategy):
    name = "long_AG_4h_v10"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    frama_period: int = 50
    wr_period: int = 35
    mfi_period: int = 40
    wr_threshold: float = -50.0
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._frama = frama(self._closes, period=self.frama_period)
        self._wr = williams_r(
            self._highs, self._lows, self._closes, period=self.wr_period,
        )
        self._mfi = mfi(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.mfi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        f = self._frama[bar_index]
        f_prev = self._frama[bar_index - 1]
        wr = self._wr[bar_index]
        m = self._mfi[bar_index]

        if any(np.isnan(v) for v in (f, f_prev, wr, m)):
            return 0.0

        # Williams %R: -100 to 0, above -50 = bullish momentum
        if f > f_prev and wr > self.wr_threshold and m > 50.0:
            strength = min(1.0, (wr + 100.0) / 50.0 * 0.4 + (m - 50.0) / 50.0 * 0.4 + 0.2)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"FRAMA({self.frama_period})", "array": self._frama, "type": "overlay"},
            {"name": f"Williams%R({self.wr_period})", "array": self._wr,
             "type": "subplot", "y_range": [-100, 0], "horizontal_lines": [-20, -50, -80]},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi,
             "type": "subplot", "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"FRAMA({self.frama_period})", datetimes, self._frama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Williams%R({self.wr_period})",
                    [self._make_subplot_trace("%R", datetimes, self._wr, color="#ab47bc")],
                    y_range=[-100, 0], horizontal_lines=[-20, -50, -80],
                ),
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#42a5f5")],
                    y_range=[0, 100], horizontal_lines=[20, 50, 80],
                ),
            ],
        }
