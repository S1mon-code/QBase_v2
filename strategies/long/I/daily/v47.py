"""MildTrendLongIDailyV47 — TSI(18,30) bullish + Klinger(20,40) > 0.

Economic logic: TSI double-smooths momentum to remove noise while preserving
trend signals. Bullish TSI (above zero or crossing up) combined with positive
Klinger oscillator confirms volume-weighted buying pressure supports the trend.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV47(TrendingStrategy):
    name = "long_I_daily_v47"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 70

    tsi_long: int = 30
    tsi_short: int = 18
    klinger_fast: int = 20
    klinger_slow: int = 40
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tsi, _ = tsi(self._closes, long_period=self.tsi_long, short_period=self.tsi_short)
        self._klinger, self._klinger_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast, slow=self.klinger_slow)

    def _generate_signal(self, bar_index: int) -> float:
        tsi_val = self._tsi[bar_index]
        tsi_prev = self._tsi[bar_index - 1] if bar_index > 0 else np.nan
        kvo = self._klinger[bar_index]

        if any(np.isnan(v) for v in [tsi_val, tsi_prev, kvo]):
            return 0.0

        tsi_bullish = tsi_val > 0.0
        tsi_rising = tsi_val > tsi_prev
        klinger_positive = kvo > 0.0

        if not ((tsi_bullish or tsi_rising) and klinger_positive):
            return 0.0

        # Scale by TSI magnitude
        tsi_score = min(0.5, max(0.0, tsi_val / 50.0)) + 0.3
        if tsi_bullish and tsi_rising:
            tsi_score += 0.15
        return min(1.0, tsi_score)

    def get_indicator_config(self):
        return [
            {"name": f"TSI({self.tsi_short},{self.tsi_long})", "array": self._tsi,
             "type": "subplot", "horizontal_lines": [0]},
            {"name": "Klinger", "array": self._klinger, "type": "subplot",
             "panel": "Klinger", "horizontal_lines": [0]},
            {"name": "Klinger Signal", "array": self._klinger_signal, "type": "subplot",
             "panel": "Klinger", "style": "dash"},
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
                    "Klinger",
                    [
                        self._make_subplot_trace("KVO", datetimes, self._klinger, color="#26a69a"),
                        self._make_subplot_trace("Signal", datetimes, self._klinger_signal,
                                                 color="#ef5350", style="dash"),
                    ],
                    horizontal_lines=[0],
                ),
            ],
        }
