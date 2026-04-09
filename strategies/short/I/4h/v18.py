"""MildTrendShortI4hV18 — HMA slope negative + Stochastic weak.

Economic logic: HMA(45) declining slope on 4H Iron Ore provides smooth,
lag-reduced bearish momentum detection. Stochastic(14,5) < 30 confirms
price is near recent lows within the downtrend.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.trend.ema import ema
from indicators.momentum.stochastic import stochastic
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV18(TrendingStrategy):
    """HMA(45) slope < 0 + Stochastic(14,5) < 30."""

    name = "short_I_4h_v18"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 60

    hma_period: int = 45
    stoch_k: int = 14
    stoch_d: int = 5
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
        self._hma = hma(self._closes, self.hma_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes, self.stoch_k, self.stoch_d,
        )

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, bar_index: int) -> float:
        h = self._hma[bar_index]
        h_prev = self._hma[bar_index - 1]
        sk = self._stoch_k[bar_index]

        if any(np.isnan(v) for v in (h, h_prev, sk)):
            return 0.0

        slope = h - h_prev
        if slope >= 0:
            return 0.0

        slope_pct = abs(slope) / h_prev if h_prev != 0 else 0.0
        signal = -(0.3 + min(0.2, slope_pct * 220))

        if sk < 30:
            signal -= min(0.15, (30 - sk) / 100.0)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay", "color": "#ff7043"},
            {"name": f"Stoch %K({self.stoch_k})", "array": self._stoch_k, "type": "subplot",
             "panel": "Stochastic", "y_range": [0, 100], "horizontal_lines": [20, 30, 80]},
            {"name": f"Stoch %D({self.stoch_d})", "array": self._stoch_d, "type": "subplot",
             "panel": "Stochastic", "color": "#ff9800"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ff7043"),
            ],
            "subplots": [
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
