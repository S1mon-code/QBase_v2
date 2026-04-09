"""MildTrendLongI4hV36 — TSI(14,26) bullish + Force Index(14) > 0.

Economic logic: TSI double-smooths momentum on 4h bars for clean trend signals.
Positive Force Index confirms volume-weighted buying pressure aligns with price
direction. Dual momentum and volume validation.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV36(TrendingStrategy):
    name = "mild_trend_long_I_4h_v36"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 50

    tsi_long: int = 26
    tsi_short: int = 14
    fi_period: int = 14
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tsi, _ = tsi(self._closes, long_period=self.tsi_long, short_period=self.tsi_short)
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        tsi_val = self._tsi[bar_index]
        tsi_prev = self._tsi[bar_index - 1] if bar_index > 0 else np.nan
        fi_val = self._fi[bar_index]

        if any(np.isnan(v) for v in [tsi_val, tsi_prev, fi_val]):
            return 0.0

        tsi_bullish = tsi_val > 0.0
        tsi_rising = tsi_val > tsi_prev
        fi_positive = fi_val > 0.0

        if not ((tsi_bullish or tsi_rising) and fi_positive):
            return 0.0

        tsi_score = min(0.5, max(0.0, tsi_val / 50.0)) + 0.3
        if tsi_bullish and tsi_rising:
            tsi_score += 0.15
        return min(1.0, tsi_score)

    def get_indicator_config(self):
        return [
            {"name": f"TSI({self.tsi_short},{self.tsi_long})", "array": self._tsi,
             "type": "subplot", "horizontal_lines": [0]},
            {"name": f"Force Index({self.fi_period})", "array": self._fi, "type": "subplot",
             "horizontal_lines": [0]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"TSI({self.tsi_short},{self.tsi_long})",
                    [self._make_subplot_trace("TSI", datetimes, self._tsi, color="#ab47bc")],
                    horizontal_lines=[0],
                ),
                self._make_subplot(
                    f"Force Index({self.fi_period})",
                    [self._make_subplot_trace("FI", datetimes, self._fi, color="#66bb6a")],
                    horizontal_lines=[0],
                ),
            ],
        }
