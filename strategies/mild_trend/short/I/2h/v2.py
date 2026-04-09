"""MildTrendShortI2hV2 — SuperTrend Bearish + MACD Negative + BSP Selling.

Economic logic: SuperTrend bearish on 2H confirms directional shift. MACD line
below zero validates momentum decay. Buying/Selling Pressure ratio below 1
signals seller dominance in volume participation.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.momentum.macd import macd
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV2(TrendingStrategy):
    """SuperTrend(40,2.5) bearish + MACD(12,35,10)<0 + BSP selling."""

    name = "mild_trend_short_I_2h_v2"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 55

    st_period: int = 40
    st_mult: float = 2.5
    macd_fast: int = 12
    macd_slow: int = 35
    chandelier_mult: float = 2.5

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
        self._macd_line, self._macd_sig, self._macd_hist = macd(
            self._closes, self.macd_fast, self.macd_slow, 10,
        )
        self._bp, self._sp, self._bsp_ratio = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes, 35,
        )

    def _generate_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        macd_val = self._macd_line[bar_index]
        ratio = self._bsp_ratio[bar_index]

        if any(np.isnan(v) for v in (st_d, macd_val)):
            return 0.0

        if st_d > 0:
            return 0.0

        signal = -0.3

        if macd_val < 0:
            macd_str = min(1.0, abs(macd_val) / 15.0)
            signal -= 0.25 * macd_str

        if not np.isnan(ratio) and ratio < 1.0:
            sell_str = min(1.0, (1.0 - ratio) * 3.0)
            signal -= 0.2 * sell_str

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"SuperTrend({self.st_period})", "array": self._st_line, "type": "overlay", "color": "#ef5350"},
            {"name": "MACD", "array": self._macd_line, "type": "subplot", "panel": "MACD", "zero_line": True},
            {"name": "MACD Hist", "array": self._macd_hist, "type": "subplot", "panel": "MACD", "style": "bar",
             "color_positive": "#26a69a", "color_negative": "#ef5350"},
            {"name": "BSP Ratio", "array": self._bsp_ratio, "type": "subplot", "horizontal_lines": [1.0]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period},{self.st_mult})", datetimes, self._st_line, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    "MACD",
                    [
                        self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#bb86fc"),
                        self._make_subplot_trace("Hist", datetimes, self._macd_hist, style="bar",
                                                 color_positive="#26a69a", color_negative="#ef5350"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "BSP Ratio",
                    [self._make_subplot_trace("Ratio", datetimes, self._bsp_ratio, color="#ffab40")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
