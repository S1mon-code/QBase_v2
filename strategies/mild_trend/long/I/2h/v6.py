"""MildTrendLongI2hV6 — LinearReg(80) + TRIX(50) + ForceIndex(40).

Economic logic: Linear regression provides a statistical trend anchor for 2H iron ore.
TRIX triple-smoothed momentum oscillator filters short-term noise effectively.
Force Index combines price change with volume for conviction measure. Signal scales
with TRIX level and force index positivity.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.trix import trix
from indicators.trend.linear_regression import linear_regression
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV6(TrendingStrategy):
    name = "mild_trend_long_I_2h_v6"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 180

    lr_period: int = 80
    trix_period: int = 50
    fi_period: int = 40
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._lr = linear_regression(self._closes, period=self.lr_period)
        self._trix_line, self._trix_signal = trix(self._closes, period=self.trix_period)
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        lr_val = self._lr[bar_index]
        tx = self._trix_line[bar_index]
        fi_val = self._fi[bar_index]
        close = self._closes[bar_index]

        if any(np.isnan(v) for v in [lr_val, tx, fi_val, close]):
            return 0.0

        above_lr = close > lr_val
        trix_positive = tx > 0.0
        fi_positive = fi_val > 0.0

        if not (above_lr and trix_positive and fi_positive):
            return 0.0

        trix_score = min(1.0, tx / 0.05) * 0.4
        return min(1.0, 0.3 + trix_score + 0.2)

    def get_indicator_config(self):
        return [
            {"name": f"LinReg({self.lr_period})", "array": self._lr, "type": "overlay"},
            {"name": "TRIX", "array": self._trix_line, "type": "subplot", "panel": "TRIX", "zero_line": True},
            {"name": "TRIX Signal", "array": self._trix_signal, "type": "subplot", "panel": "TRIX", "style": "dash"},
            {"name": f"Force Index({self.fi_period})", "array": self._fi, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"LinReg({self.lr_period})", datetimes, self._lr, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "TRIX",
                    [
                        self._make_subplot_trace("TRIX", datetimes, self._trix_line, color="#bb86fc"),
                        self._make_subplot_trace("Signal", datetimes, self._trix_signal, color="#ff8a80", style="dash"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "Force Index",
                    [self._make_subplot_trace("FI", datetimes, self._fi, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
