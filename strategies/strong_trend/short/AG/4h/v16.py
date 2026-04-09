"""StrongTrendShortAG4hV16 — Aroon(22) down > 72 + OBV below SMA(40).

Economic logic: Aroon(22) Down above 72 on 4H silver signals recent new lows
within the lookback — strong bearish momentum. OBV below SMA(40) confirms
persistent distribution over a longer horizon. The dual confirmation prevents
false shorts from isolated spike lows.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.trend.ema import ema
from indicators.trend.sma import sma
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV16(TrendingStrategy):
    """Aroon(22) Down > 72 + OBV < SMA(40)."""

    name = "strong_trend_short_AG_4h_v16"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    aroon_period: int = 22
    obv_sma_period: int = 40
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._aroon_up, self._aroon_down, _ = aroon(self._highs, self._lows, self.aroon_period)
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, period=self.obv_sma_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        ad = self._aroon_down[i]
        au = self._aroon_up[i]
        ov = self._obv[i]
        os_ = self._obv_sma[i]

        if any(np.isnan(v) for v in (ad, au, ov, os_)):
            return 0.0

        if ad <= 72.0 or ov >= os_:
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
            {"name": "OBV", "array": self._obv, "panel": "OBV"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma, "panel": "OBV"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(f"Aroon({self.aroon_period})", [
                    self._make_subplot_trace("Up", datetimes, self._aroon_up, color="#66bb6a"),
                    self._make_subplot_trace("Down", datetimes, self._aroon_down, color="#ef5350"),
                ], horizontal_lines=[25, 72], y_range=[0, 100]),
                self._make_subplot("OBV", [
                    self._make_subplot_trace("OBV", datetimes, self._obv, color="#7e57c2"),
                    self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ff9800"),
                ]),
            ],
        }
