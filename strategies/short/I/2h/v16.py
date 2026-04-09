"""MildTrendShortI2hV16 — Aroon down dominant + OBV below SMA.

Economic logic: Aroon(20) down > 70 on 2H Iron Ore indicates recent lows
are dominating the lookback window. OBV falling below its SMA(30) validates
sustained distribution, showing volume-weighted selling outpacing buying.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.trend.ema import ema
from indicators.trend.sma import sma
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV16(TrendingStrategy):
    """Aroon(20) down > 70 + OBV < SMA(30)."""

    name = "short_I_2h_v16"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    aroon_period: int = 20
    obv_sma_period: int = 30
    aroon_threshold: float = 70.0
    chandelier_mult: float = 3.0

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
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, self.obv_sma_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, bar_index: int) -> float:
        a_down = self._aroon_down[bar_index]
        a_up = self._aroon_up[bar_index]
        ov = self._obv[bar_index]
        os_ = self._obv_sma[bar_index]

        if any(np.isnan(v) for v in (a_down, a_up, os_)):
            return 0.0

        if a_down <= self.aroon_threshold:
            return 0.0

        dominance = (a_down - a_up) / 100.0
        signal = -(0.3 + min(0.2, dominance * 0.4))

        if ov < os_:
            signal -= 0.15

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
            {"name": "OBV", "array": self._obv, "type": "subplot"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma,
             "type": "subplot", "panel": "OBV", "color": "#ff9800"},
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
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv, color="#2196f3"),
                        self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ff9800"),
                    ],
                ),
            ],
        }
