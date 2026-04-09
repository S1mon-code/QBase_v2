"""StrongTrendShortAG1hV16 — Aroon(12) down > 75 + OI momentum(10) < 0.

Economic logic: Aroon(12) Down above 75 on 1H silver means a new low was hit
within the last 3 bars — strong intraday bearish momentum. OI momentum negative
confirms open interest is declining or shifting bearish, signaling that new
shorts are being opened or longs are liquidating.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.trend.ema import ema
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV16(TrendingStrategy):
    """Aroon(12) Down > 75 + OI momentum(10) < 0."""

    name = "short_AG_1h_v16"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 20

    aroon_period: int = 12
    oi_period: int = 10
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._aroon_up, self._aroon_down, _ = aroon(self._highs, self._lows, self.aroon_period)
        self._oi_mom = oi_momentum(self._oi, self.oi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, i: int) -> float:
        ad = self._aroon_down[i]
        au = self._aroon_up[i]
        om = self._oi_mom[i]

        if any(np.isnan(v) for v in (ad, au, om)):
            return 0.0

        if ad <= 75.0 or om >= 0:
            return 0.0

        strength = -0.45
        if ad > 90.0:
            strength -= 0.20
        if au < 25.0:
            strength -= 0.20
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
            {"name": f"OI Mom({self.oi_period})", "array": self._oi_mom, "panel": "OI", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(f"Aroon({self.aroon_period})", [
                    self._make_subplot_trace("Up", datetimes, self._aroon_up, color="#66bb6a"),
                    self._make_subplot_trace("Down", datetimes, self._aroon_down, color="#ef5350"),
                ], horizontal_lines=[25, 75], y_range=[0, 100]),
                self._make_subplot(f"OI Momentum({self.oi_period})", [
                    self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#42a5f5"),
                ], zero_line=True),
            ],
        }
