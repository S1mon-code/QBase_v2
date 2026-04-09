"""MildTrendLongI2hV37 — CCI(16) > 0 + OI momentum(14) > 0.

Economic logic: CCI on 2h timeframe captures medium-term price deviations. Positive CCI
shows buying pressure. OI momentum confirms new long positions are being opened.
Dual confirmation from price and open interest positioning.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV37(TrendingStrategy):
    name = "long_I_2h_v37"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 30

    cci_period: int = 16
    oi_period: int = 14
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        self._oi_mom = oi_momentum(self._oi, period=self.oi_period)

    def _generate_signal(self, bar_index: int) -> float:
        cci_val = self._cci[bar_index]
        oi_val = self._oi_mom[bar_index]

        if any(np.isnan(v) for v in [cci_val, oi_val]):
            return 0.0

        cci_positive = cci_val > 0.0
        oi_bullish = oi_val > 0.0

        if not (cci_positive and oi_bullish):
            return 0.0

        cci_score = min(0.5, max(0.0, cci_val / 400.0)) + 0.3
        oi_boost = min(0.2, max(0.0, oi_val * 0.5))
        return min(1.0, cci_score + oi_boost)

    def get_indicator_config(self):
        return [
            {"name": f"CCI({self.cci_period})", "array": self._cci, "type": "subplot",
             "horizontal_lines": [0, 100, -100]},
            {"name": f"OI Mom({self.oi_period})", "array": self._oi_mom, "type": "subplot",
             "horizontal_lines": [0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"CCI({self.cci_period})",
                    [self._make_subplot_trace("CCI", datetimes, self._cci, color="#42a5f5")],
                    horizontal_lines=[0, 100, -100],
                ),
                self._make_subplot(
                    f"OI Momentum({self.oi_period})",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom,
                                             color="#66bb6a")],
                    horizontal_lines=[0],
                ),
            ],
        }
