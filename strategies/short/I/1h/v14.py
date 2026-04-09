"""MildTrendShortI1hV14 — Donchian lower break + Force Index negative.

Economic logic: Price breaking below Donchian(20) lower channel on 1H Iron Ore
signals a new short-term low. Force Index(8) < 0 confirms volume-weighted
selling momentum, ensuring the breakdown has conviction behind it.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.trend.ema import ema
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV14(TrendingStrategy):
    """Donchian(20) lower break + Force Index(8) < 0."""

    name = "short_I_1h_v14"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    dc_period: int = 20
    fi_period: int = 8
    chandelier_mult: float = 2.8

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
        self._fi = force_index(self._closes, self._volumes, self.fi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        price = self._closes[bar_index]
        dc_low = self._dc_lower[bar_index]
        dc_up = self._dc_upper[bar_index]
        fi = self._fi[bar_index]

        if any(np.isnan(v) for v in (dc_low, dc_up, fi)):
            return 0.0

        if price > dc_low:
            return 0.0

        dc_range = dc_up - dc_low
        if dc_range <= 0:
            return 0.0

        penetration = (dc_low - price) / dc_range
        signal = -(0.35 + min(0.2, penetration * 3.0))

        if fi < 0:
            signal -= min(0.1, abs(fi) / 1e6 * 0.1)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay", "color": "#66bb6a"},
            {"name": f"DC Lower({self.dc_period})", "array": self._dc_lower, "type": "overlay", "color": "#ef5350"},
            {"name": f"Force Index({self.fi_period})", "array": self._fi, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, color="#66bb6a"),
                self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Force Index({self.fi_period})",
                    [self._make_subplot_trace("FI", datetimes, self._fi, color="#ef5350")],
                    zero_line=True,
                ),
            ],
        }
