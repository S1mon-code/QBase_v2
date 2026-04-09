"""MildTrendShortI1hV16 — Aroon down dominant + Volume spike.

Economic logic: Aroon(14) down > 70 on 1H Iron Ore indicates recent lows
dominate the lookback window. Volume spike(10) > 1.5 during the bearish
phase confirms panic selling or capitulation, adding conviction to the
short signal.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.trend.ema import ema
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV16(TrendingStrategy):
    """Aroon(14) down > 70 + Volume spike(10)."""

    name = "mild_trend_short_I_1h_v16"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 25

    aroon_period: int = 14
    vol_spike_period: int = 10
    aroon_threshold: float = 70.0
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
        self._aroon_up, self._aroon_down, _ = aroon(self._highs, self._lows, self.aroon_period)
        self._vol_spike = volume_spike(self._volumes, self.vol_spike_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        a_down = self._aroon_down[bar_index]
        a_up = self._aroon_up[bar_index]
        vs = self._vol_spike[bar_index]

        if any(np.isnan(v) for v in (a_down, a_up, vs)):
            return 0.0

        if a_down <= self.aroon_threshold:
            return 0.0

        dominance = (a_down - a_up) / 100.0
        signal = -(0.3 + min(0.2, dominance * 0.4))

        if vs > 1.5:
            signal -= min(0.15, (vs - 1.5) * 0.1)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"Aroon Up({self.aroon_period})", "array": self._aroon_up, "type": "subplot",
             "panel": "Aroon", "color": "#66bb6a"},
            {"name": f"Aroon Down({self.aroon_period})", "array": self._aroon_down, "type": "subplot",
             "panel": "Aroon", "color": "#ef5350"},
            {"name": "Vol Spike", "array": self._vol_spike, "type": "subplot", "horizontal_lines": [1.5]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"Aroon({self.aroon_period})",
                    [
                        self._make_subplot_trace("Up", datetimes, self._aroon_up, color="#66bb6a"),
                        self._make_subplot_trace("Down", datetimes, self._aroon_down, color="#ef5350"),
                    ],
                    horizontal_lines=[self.aroon_threshold], y_range=[0, 100],
                ),
                self._make_subplot(
                    "Volume Spike",
                    [self._make_subplot_trace("Spike", datetimes, self._vol_spike, color="#26c6da")],
                    horizontal_lines=[1.5],
                ),
            ],
        }
