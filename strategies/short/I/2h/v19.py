"""MildTrendShortI2hV19 — Linear regression slope + Stochastic weak.

Economic logic: Linear regression slope(25) < 0 on 2H Iron Ore quantifies the
statistical downtrend. Stochastic(14,5) < 30 confirms price is near recent
lows within that downtrend, signaling continued bearish pressure.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression_slope
from indicators.trend.ema import ema
from indicators.momentum.stochastic import stochastic
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV19(TrendingStrategy):
    """LinReg slope(25) < 0 + Stochastic(14,5) < 30."""

    name = "short_I_2h_v19"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 40

    lr_period: int = 25
    stoch_k: int = 14
    stoch_d: int = 5
    chandelier_mult: float = 3.0

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
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes, self.stoch_k, self.stoch_d,
        )

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, bar_index: int) -> float:
        lr = self._lr_slope[bar_index]
        sk = self._stoch_k[bar_index]
        sd = self._stoch_d[bar_index]

        if any(np.isnan(v) for v in (lr, sk, sd)):
            return 0.0

        if lr >= 0:
            return 0.0

        signal = -(0.3 + min(0.2, abs(lr) * 0.8))

        if sk < 30:
            signal -= min(0.15, (30 - sk) / 100.0)

        if sd < 30:
            signal -= 0.05

        return max(-0.7, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"LR Slope({self.lr_period})", "array": self._lr_slope, "type": "subplot", "zero_line": True},
            {"name": f"Stoch %K({self.stoch_k})", "array": self._stoch_k, "type": "subplot",
             "panel": "Stochastic", "y_range": [0, 100], "horizontal_lines": [20, 30, 80]},
            {"name": f"Stoch %D({self.stoch_d})", "array": self._stoch_d, "type": "subplot",
             "panel": "Stochastic", "color": "#ff9800"},
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
                    "Stochastic",
                    [
                        self._make_subplot_trace("%K", datetimes, self._stoch_k, color="#bb86fc"),
                        self._make_subplot_trace("%D", datetimes, self._stoch_d, color="#ff9800"),
                    ],
                    horizontal_lines=[20, 30, 80], y_range=[0, 100],
                ),
            ],
        }
