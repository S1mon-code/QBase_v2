"""MildTrendShortI4hV19 — Linear regression slope + TRIX bearish.

Economic logic: Linear regression slope(30) < 0 on 4H Iron Ore quantifies
the statistical downtrend. TRIX(16) < 0 adds triple-smoothed momentum
confirmation, ensuring the bearish move is persistent rather than noise.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression_slope
from indicators.trend.ema import ema
from indicators.momentum.trix import trix
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV19(TrendingStrategy):
    """LinReg slope(30) < 0 + TRIX(16) < 0."""

    name = "short_I_4h_v19"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 55

    lr_period: int = 30
    trix_period: int = 16
    chandelier_mult: float = 3.4

    def on_init_arrays(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        volumes: np.ndarray,
        oi: np.ndarray,
        datetimes: np.ndarray,
    ) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._lr_slope = linear_regression_slope(self._closes, self.lr_period)
        self._trix, self._trix_signal = trix(self._closes, self.trix_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, bar_index: int) -> float:
        lr = self._lr_slope[bar_index]
        t = self._trix[bar_index]

        if any(np.isnan(v) for v in (lr, t)):
            return 0.0

        if lr >= 0:
            return 0.0

        # Base from slope magnitude
        signal = -(0.3 + min(0.2, abs(lr) * 0.6))

        # TRIX confirmation
        if t < 0:
            signal -= min(0.15, abs(t) * 50)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"LR Slope({self.lr_period})", "array": self._lr_slope, "type": "subplot", "zero_line": True},
            {"name": f"TRIX({self.trix_period})", "array": self._trix, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"LR Slope({self.lr_period})",
                    [self._make_subplot_trace("Slope", datetimes, self._lr_slope, color="#42a5f5")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"TRIX({self.trix_period})",
                    [self._make_subplot_trace("TRIX", datetimes, self._trix, color="#ef5350")],
                    zero_line=True,
                ),
            ],
        }
