"""MildTrendShortIDailyV5 — Linear Regression Slope + CCI + Klinger.

Economic logic: Linear regression slope over 60 bars turning negative
confirms a structural downtrend. CCI below -30 validates price weakness
relative to its statistical mean. Negative Klinger volume oscillator
adds distribution pressure confirmation.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression_slope
from indicators.momentum.cci import cci
from indicators.volume.klinger import klinger
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV5(TrendingStrategy):
    """Linear Reg(60) slope < 0 + CCI(25) < -30 + Klinger(20) < 0."""

    name = "short_I_daily_v5"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    # Optimizable parameters (<=5 including chandelier_mult)
    lr_period: int = 60
    cci_period: int = 25
    cci_threshold: float = -30.0
    klinger_fast: int = 20
    chandelier_mult: float = 3.5

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
        self._cci = cci(self._highs, self._lows, self._closes, self.cci_period)
        self._kvo, self._kvo_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast, slow=55, signal=13,
        )

        # Pre-smooth raw signal
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        slope = self._lr_slope[bar_index]
        cci_val = self._cci[bar_index]
        kvo_val = self._kvo[bar_index]

        if any(np.isnan(v) for v in (slope, cci_val, kvo_val)):
            return 0.0

        # LR slope must be negative
        if slope >= 0:
            return 0.0

        # CCI must be below threshold
        if cci_val >= self.cci_threshold:
            return 0.0

        # Slope magnitude drives base signal
        slope_strength = min(abs(slope) / 3.0, 1.0)
        base = -0.3 - slope_strength * 0.3  # -0.3 to -0.6

        # Klinger confirmation
        if kvo_val < 0:
            base -= 0.1

        return np.clip(base, -0.7, 0.0)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"LR Slope({self.lr_period})", "array": self._lr_slope,
             "type": "subplot", "zero_line": True, "color": "#ffab40"},
            {"name": f"CCI({self.cci_period})", "array": self._cci,
             "type": "subplot", "zero_line": True,
             "horizontal_lines": [self.cci_threshold], "color": "#2196f3"},
            {"name": f"Klinger({self.klinger_fast})", "array": self._kvo,
             "type": "subplot", "zero_line": True, "color": "#ab47bc"},
            {"name": "Klinger Signal", "array": self._kvo_signal,
             "type": "subplot", "panel": f"Klinger({self.klinger_fast})",
             "color": "#ff9800"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"LR Slope({self.lr_period})",
                    [self._make_subplot_trace("Slope", datetimes, self._lr_slope, color="#ffab40")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"CCI({self.cci_period})",
                    [self._make_subplot_trace("CCI", datetimes, self._cci, color="#2196f3")],
                    zero_line=True, horizontal_lines=[self.cci_threshold],
                ),
                self._make_subplot(
                    f"Klinger({self.klinger_fast})",
                    [
                        self._make_subplot_trace("KVO", datetimes, self._kvo, color="#ab47bc"),
                        self._make_subplot_trace("Signal", datetimes, self._kvo_signal, color="#ff9800"),
                    ],
                    zero_line=True,
                ),
            ],
        }
