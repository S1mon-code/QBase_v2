"""StrongTrendShortAG2hV16 — Aroon(18) down > 70 + Volume spike(12).

Economic logic: Aroon(18) Down above 70 on 2H silver signals recent new lows
within the lookback window. Volume spike confirms institutional selling — large
volume bursts during the downtrend validate that the move has real participation.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.trend.ema import ema
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV16(TrendingStrategy):
    """Aroon(18) Down > 70 + Volume spike(12)."""

    name = "strong_trend_short_AG_2h_v16"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 25

    aroon_period: int = 18
    vs_period: int = 12
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._aroon_up, self._aroon_down, _ = aroon(self._highs, self._lows, self.aroon_period)
        self._vs = volume_spike(self._volumes, self.vs_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        ad = self._aroon_down[i]
        au = self._aroon_up[i]
        vs = self._vs[i]

        if any(np.isnan(v) for v in (ad, au, vs)):
            return 0.0

        if ad <= 70.0:
            return 0.0

        strength = -0.40
        if vs > 1.5:
            strength -= 0.25
        if ad > 90.0:
            strength -= 0.15
        if au < 25.0:
            strength -= 0.10
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": "Aroon Up", "array": self._aroon_up, "panel": "Aroon", "color": "#66bb6a"},
            {"name": "Aroon Down", "array": self._aroon_down, "panel": "Aroon", "color": "#ef5350"},
            {"name": f"Vol Spike({self.vs_period})", "array": self._vs, "panel": "VS"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(f"Aroon({self.aroon_period})", [
                    self._make_subplot_trace("Up", datetimes, self._aroon_up, color="#66bb6a"),
                    self._make_subplot_trace("Down", datetimes, self._aroon_down, color="#ef5350"),
                ], horizontal_lines=[25, 70], y_range=[0, 100]),
                self._make_subplot(f"Volume Spike({self.vs_period})", [
                    self._make_subplot_trace("Spike", datetimes, self._vs, color="#42a5f5"),
                ], horizontal_lines=[1.5]),
            ],
        }
