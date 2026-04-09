"""MildTrendShortIDailyV10 — TEMA Slope + Aroon Oscillator + OBV Trend.

Economic logic: TEMA(50) slope turning negative provides triple-smoothed
trend confirmation with minimal lag. Aroon Oscillator below -20 shows
bearish dominance in recent high/low positioning. OBV declining below
its SMA confirms volume-based distribution pressure.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.tema import tema
from indicators.trend.aroon import aroon
from indicators.volume.obv import obv
from indicators.trend.sma import sma
from indicators.trend.linear_regression import linear_regression_slope
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV10(TrendingStrategy):
    """TEMA(50) slope < 0 + Aroon Osc(30) < -20 + OBV trend down."""

    name = "mild_trend_short_I_daily_v10"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    # Optimizable parameters (<=5 including chandelier_mult)
    tema_period: int = 50
    aroon_period: int = 30
    obv_sma_period: int = 40
    aroon_threshold: float = -20.0
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
        self._tema_line = tema(self._closes, self.tema_period)
        self._tema_slope = linear_regression_slope(self._tema_line, 10)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, self.aroon_period,
        )
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, self.obv_sma_period)

        # Pre-smooth raw signal
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        slope = self._tema_slope[bar_index]
        aroon_o = self._aroon_osc[bar_index]
        obv_val = self._obv[bar_index]
        obv_sma_val = self._obv_sma[bar_index]

        if any(np.isnan(v) for v in (slope, aroon_o, obv_sma_val)):
            return 0.0

        # TEMA slope must be negative
        if slope >= 0:
            return 0.0

        # Aroon Oscillator must be below threshold
        if aroon_o >= self.aroon_threshold:
            return 0.0

        # Aroon magnitude drives base signal
        aroon_strength = min(abs(aroon_o) / 100.0, 1.0)
        base = -0.3 - aroon_strength * 0.3  # -0.3 to -0.6

        # OBV trend confirmation
        if obv_val < obv_sma_val:
            base -= 0.1

        return np.clip(base, -0.7, 0.0)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema_line,
             "type": "overlay", "color": "#ffab40"},
            {"name": f"Aroon Osc({self.aroon_period})", "array": self._aroon_osc,
             "type": "subplot", "zero_line": True,
             "horizontal_lines": [self.aroon_threshold],
             "y_range": [-100, 100], "color": "#2196f3"},
            {"name": "OBV", "array": self._obv,
             "type": "subplot", "color": "#ab47bc"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma,
             "type": "subplot", "panel": "OBV", "color": "#ff9800"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Aroon Osc({self.aroon_period})",
                    [self._make_subplot_trace("Aroon Osc", datetimes, self._aroon_osc, color="#2196f3")],
                    zero_line=True, horizontal_lines=[self.aroon_threshold],
                    y_range=[-100, 100],
                ),
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv, color="#ab47bc"),
                        self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ff9800"),
                    ],
                ),
            ],
        }
