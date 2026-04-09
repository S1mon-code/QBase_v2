"""StrongTrendLongAG4hV3 — HMA(40) + Fisher(30) + CMF(40).

Economic logic: HMA provides fast, lag-free trend detection on AG 4H bars.
Fisher Transform normalizes price extremes for cleaner momentum signals.
CMF confirms volume-weighted money flow direction.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG4hV3(TrendingStrategy):
    name = "strong_trend_long_AG_4h_v3"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    hma_period: int = 40
    fisher_period: int = 30
    cmf_period: int = 40
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, period=self.hma_period)
        self._fisher, self._fisher_sig = fisher_transform(
            self._highs, self._lows, period=self.fisher_period,
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1]
        fish = self._fisher[bar_index]
        c = self._cmf[bar_index]

        if any(np.isnan(v) for v in (h, h_prev, fish, c)):
            return 0.0

        if h > h_prev and fish > 0.0 and c > 0.0:
            strength = min(1.0, abs(fish) / 2.5 * 0.5 + c * 2.0)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": f"Fisher({self.fisher_period})", "array": self._fisher,
             "type": "subplot", "zero_line": True},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf,
             "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Fisher({self.fisher_period})",
                    [self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._fisher_sig, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#66bb6a")],
                    zero_line=True,
                ),
            ],
        }
