"""StrongTrendLongAG2hV4 — KAMA(55) + TSI(35,18) + CMF(50).

Economic logic: KAMA adapts smoothing to AG's 2H efficiency ratio. TSI
double-smoothed momentum filters Silver's choppy intraday noise. CMF
confirms volume-weighted money flow direction.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.momentum.tsi import tsi
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG2hV4(TrendingStrategy):
    name = "long_AG_2h_v4"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    kama_period: int = 55
    tsi_long: int = 35
    tsi_short: int = 18
    cmf_period: int = 50
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=self.tsi_short,
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1]
        t = self._tsi_line[bar_index]
        c = self._cmf[bar_index]

        if any(np.isnan(v) for v in (k, k_prev, t, c)):
            return 0.0

        if k > k_prev and t > 0.0 and c > 0.0:
            strength = min(1.0, t / 25.0 * 0.5 + c * 2.0 + 0.2)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": f"TSI({self.tsi_long},{self.tsi_short})", "array": self._tsi_line,
             "type": "subplot", "zero_line": True},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf,
             "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"TSI({self.tsi_long},{self.tsi_short})",
                    [self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._tsi_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#66bb6a")],
                    zero_line=True,
                ),
            ],
        }
