"""MildTrendShortIDailyV14 — Donchian lower channel break + ADX trending.

Economic logic: Price breaking below the Donchian(50) lower channel on daily
Iron Ore signals a new low in the medium-term range, indicating bearish
momentum. ADX(25) > 20 confirms directional movement, filtering out
range-bound false breakdowns.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.trend.adx import adx
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV14(TrendingStrategy):
    """Donchian(50) lower break + ADX(25) > 20."""

    name = "mild_trend_short_I_daily_v14"
    horizon = "slow"
    direction = "short"
    signal_dimensions = ["momentum"]
    warmup: int = 70

    dc_period: int = 50
    adx_period: int = 25
    adx_threshold: float = 20.0
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
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, self.dc_period,
        )
        self._adx = adx(self._highs, self._lows, self._closes, self.adx_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        price = self._closes[bar_index]
        dc_low = self._dc_lower[bar_index]
        adx_val = self._adx[bar_index]

        if any(np.isnan(v) for v in (dc_low, adx_val)):
            return 0.0

        if adx_val <= self.adx_threshold:
            return 0.0

        # Price at or below lower channel
        if price > dc_low:
            return 0.0

        # Signal strength from penetration depth
        dc_range = self._dc_upper[bar_index] - dc_low
        if dc_range <= 0:
            return 0.0
        penetration = (dc_low - price) / dc_range
        signal = -(0.35 + min(0.25, penetration * 2.0))

        # ADX strength bonus
        if adx_val > 30:
            signal -= 0.1

        return max(-0.7, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay", "color": "#66bb6a"},
            {"name": f"DC Lower({self.dc_period})", "array": self._dc_lower, "type": "overlay", "color": "#ef5350"},
            {"name": f"ADX({self.adx_period})", "array": self._adx, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [self.adx_threshold]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, color="#66bb6a"),
                self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    f"ADX({self.adx_period})",
                    [self._make_subplot_trace("ADX", datetimes, self._adx, color="#bb86fc")],
                    horizontal_lines=[self.adx_threshold], y_range=[0, 100],
                ),
            ],
        }
