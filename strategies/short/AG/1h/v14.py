"""StrongTrendShortAG1hV14 — Bollinger(15,2.0) below middle + OBV declining(12).

Economic logic: Price below the Bollinger middle band on 1H silver signals
bearish bias. OBV below its SMA(12) confirms short-term distribution — volume
is flowing out. The shorter OBV SMA adapts to the faster intraday dynamics.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.sma import sma
from indicators.volatility.bollinger import bollinger_bands
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV14(TrendingStrategy):
    """Bollinger(15,2.0) price < middle + OBV < SMA(12)."""

    name = "short_AG_1h_v14"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 20

    bb_period: int = 15
    bb_std: float = 2.0
    obv_sma_period: int = 12
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._bb_upper, self._bb_middle, self._bb_lower = bollinger_bands(
            self._closes, period=self.bb_period, num_std=self.bb_std,
        )
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, period=self.obv_sma_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, i: int) -> float:
        c = self._closes[i]
        mid = self._bb_middle[i]
        lower = self._bb_lower[i]
        ov = self._obv[i]
        os_ = self._obv_sma[i]

        if any(np.isnan(v) for v in (c, mid, lower, ov, os_)):
            return 0.0

        if c >= mid or ov >= os_:
            return 0.0

        bb_range = mid - lower
        if bb_range <= 0:
            return 0.0
        depth = min(1.0, (mid - c) / bb_range)

        strength = -(0.35 + depth * 0.40)
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": "BB Upper", "array": self._bb_upper, "type": "overlay"},
            {"name": "BB Middle", "array": self._bb_middle, "type": "overlay"},
            {"name": "BB Lower", "array": self._bb_lower, "type": "overlay"},
            {"name": "OBV", "array": self._obv, "panel": "OBV"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma, "panel": "OBV"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("BB Upper", datetimes, self._bb_upper, color="#ef5350"),
                self._make_overlay("BB Middle", datetimes, self._bb_middle, color="#ffab40"),
                self._make_overlay("BB Lower", datetimes, self._bb_lower, color="#66bb6a"),
            ],
            "subplots": [
                self._make_subplot("OBV", [
                    self._make_subplot_trace("OBV", datetimes, self._obv, color="#7e57c2"),
                    self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ff9800"),
                ]),
            ],
        }
