"""StrongTrendLongAG1hV3 — SuperTrend(25,2.8) + Fisher(18) + ForceIndex(18).

Economic logic: SuperTrend with moderate multiplier catches AG's 1H breakouts.
Fisher Transform normalizes price for clean signal detection. Force Index
combines price change and volume for Elder-style trend strength confirmation.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG1hV3(TrendingStrategy):
    name = "strong_trend_long_AG_1h_v3"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    st_period: int = 25
    st_mult: float = 2.8
    fisher_period: int = 18
    force_period: int = 18
    chandelier_mult: float = 2.8

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        self._fisher, self._fisher_sig = fisher_transform(
            self._highs, self._lows, period=self.fisher_period,
        )
        self._force = force_index(self._closes, self._volumes, period=self.force_period)

    def _generate_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        fish = self._fisher[bar_index]
        fi = self._force[bar_index]

        if any(np.isnan(v) for v in (st_d, fish, fi)):
            return 0.0

        if st_d > 0 and fish > 0.0 and fi > 0.0:
            strength = min(1.0, abs(fish) / 2.0 * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"SuperTrend({self.st_period},{self.st_mult})",
             "array": self._st_line, "type": "overlay"},
            {"name": f"Fisher({self.fisher_period})", "array": self._fisher,
             "type": "subplot", "zero_line": True},
            {"name": f"ForceIndex({self.force_period})", "array": self._force,
             "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period},{self.st_mult})",
                                   datetimes, self._st_line, color="#ff7043"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Fisher({self.fisher_period})",
                    [self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._fisher_sig, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"ForceIndex({self.force_period})",
                    [self._make_subplot_trace("Force", datetimes, self._force, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
