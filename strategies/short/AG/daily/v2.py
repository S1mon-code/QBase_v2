"""StrongTrendShortAGDailyV2 — SuperTrend Bear + MACD Negative + Twiggs Negative.

Economic logic: SuperTrend bearish direction confirms the macro downtrend.
MACD below zero validates momentum is negative. Twiggs Money Flow < 0 shows
institutional money is flowing out — a triple confirmation for silver shorts.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.supertrend import supertrend
from indicators.volume.twiggs import twiggs_money_flow
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV2(TrendingStrategy):
    """SuperTrend bear + MACD negative + Twiggs outflow.

    Signal logic:
        ST direction bearish AND MACD_line < 0 AND Twiggs < 0 -> -0.9
        ST direction bearish AND MACD_line < 0 -> -0.5
        else -> 0.0
    """

    name = "short_AG_daily_v2"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 200

    st_period: int = 100
    st_mult: float = 4.5
    twiggs_period: int = 100
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        # MACD with wide daily params
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            self._closes, fast=60, slow=140, signal=45,
        )
        self._twiggs = twiggs_money_flow(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.twiggs_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        ml = self._macd_line[bar_index]
        tw = self._twiggs[bar_index]

        if np.isnan(st_d) or np.isnan(ml):
            return 0.0

        if st_d > 0:  # bullish
            return 0.0

        if ml < 0 and (not np.isnan(tw) and tw < 0):
            return -0.9
        if ml < 0:
            return -0.5
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"SuperTrend({self.st_period})", "array": self._st_line, "type": "overlay"},
            {"name": "MACD Line", "array": self._macd_line, "panel": "MACD"},
            {"name": "MACD Signal", "array": self._macd_signal, "panel": "MACD"},
            {"name": "MACD Hist", "array": self._macd_hist, "style": "bar", "panel": "MACD",
             "color_positive": "#26a69a", "color_negative": "#ef5350"},
            {"name": f"Twiggs({self.twiggs_period})", "array": self._twiggs, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"SuperTrend({self.st_period})", datetimes, self._st_line, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot("MACD", [
                self._make_subplot_trace("MACD Line", datetimes, self._macd_line, color="#42a5f5"),
                self._make_subplot_trace("MACD Signal", datetimes, self._macd_signal, color="#ff7043"),
                self._make_subplot_trace("MACD Hist", datetimes, self._macd_hist, style="bar",
                                         color_positive="#26a69a", color_negative="#ef5350"),
            ], zero_line=True),
            self._make_subplot(f"Twiggs({self.twiggs_period})", [
                self._make_subplot_trace("Twiggs", datetimes, self._twiggs, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
