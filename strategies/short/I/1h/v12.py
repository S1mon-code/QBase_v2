"""MildTrendShortI1hV12 — SuperTrend bearish + OBV declining.

Economic logic: SuperTrend(8,2.5) flipping bearish on 1H Iron Ore captures
fast regime shifts. OBV declining below its SMA(15) confirms that volume-
weighted selling pressure is sustained, not just a single-bar spike.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.trend.ema import ema
from indicators.trend.sma import sma
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV12(TrendingStrategy):
    """SuperTrend(8,2.5) bearish + OBV < SMA(15)."""

    name = "short_I_1h_v12"
    horizon = "fast"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 25

    st_period: int = 8
    st_mult: float = 2.5
    obv_sma_period: int = 15
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
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes, self.st_period, self.st_mult,
        )
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, self.obv_sma_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        price = self._closes[bar_index]
        st_val = self._st_line[bar_index]
        ov = self._obv[bar_index]
        os_ = self._obv_sma[bar_index]

        if any(np.isnan(v) for v in (st_d, st_val, os_)):
            return 0.0

        if st_d >= 0:
            return 0.0

        dist = (st_val - price) / price if price != 0 else 0.0
        signal = -(0.3 + min(0.2, dist * 5.0))

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
            {"name": f"SuperTrend({self.st_period},{self.st_mult})", "array": self._st_line,
             "type": "overlay", "color": "#ef5350"},
            {"name": "OBV", "array": self._obv, "type": "subplot"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma,
             "type": "subplot", "panel": "OBV", "color": "#ff9800"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period},{self.st_mult})", datetimes, self._st_line, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv, color="#2196f3"),
                        self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ff9800"),
                    ],
                ),
            ],
        }
