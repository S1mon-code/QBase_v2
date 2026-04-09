"""StrongTrendLongAG2hV2 — SuperTrend(50,3.0) + ROC(35) + Twiggs(50).

Economic logic: SuperTrend with 3.0x multiplier captures AG's 2H volatility
trends. Rate of Change provides raw momentum magnitude. Twiggs Money Flow
confirms volume-weighted accumulation.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.momentum.roc import rate_of_change
from indicators.volume.twiggs import twiggs_money_flow
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG2hV2(TrendingStrategy):
    name = "strong_trend_long_AG_2h_v2"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    st_period: int = 50
    st_mult: float = 3.0
    roc_period: int = 35
    twiggs_period: int = 50
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._twiggs = twiggs_money_flow(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.twiggs_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        roc = self._roc[bar_index]
        tw = self._twiggs[bar_index]

        if any(np.isnan(v) for v in (st_d, roc, tw)):
            return 0.0

        if st_d > 0 and roc > 0.0 and tw > 0.0:
            strength = min(1.0, roc / 4.0 * 0.4 + tw * 2.0 + 0.2)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"SuperTrend({self.st_period},{self.st_mult})",
             "array": self._st_line, "type": "overlay"},
            {"name": f"ROC({self.roc_period})", "array": self._roc,
             "type": "subplot", "zero_line": True},
            {"name": f"Twiggs({self.twiggs_period})", "array": self._twiggs,
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
                    f"ROC({self.roc_period})",
                    [self._make_subplot_trace("ROC", datetimes, self._roc, color="#ab47bc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"Twiggs({self.twiggs_period})",
                    [self._make_subplot_trace("Twiggs", datetimes, self._twiggs, color="#66bb6a")],
                    zero_line=True,
                ),
            ],
        }
