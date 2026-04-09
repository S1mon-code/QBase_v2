"""StrongTrendShortAGDailyV7 — Donchian Breakdown + Schaff Bear + BSP Selling.

Economic logic: Price breaking below Donchian lower band signals new lows being
made. Schaff Trend Cycle < 25 confirms bearish momentum regime. Selling pressure
exceeding buying pressure validates the move.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.trend.donchian import donchian
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV7(TrendingStrategy):
    """Donchian breakdown + Schaff bearish + selling pressure dominant.

    Signal logic:
        close < DC_lower AND STC < 25 AND sell_pressure > buy_pressure -> -0.90
        close < DC_middle AND STC < 40 -> -0.45
        else -> 0.0
    """

    name = "strong_trend_short_AG_daily_v7"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 150

    dc_period: int = 100
    schaff_period: int = 100
    bsp_period: int = 80
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.dc_period,
        )
        self._stc = schaff_trend_cycle(
            self._closes, period=self.schaff_period, fast=60, slow=30,
        )
        self._buy_p, self._sell_p, self._ratio = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.bsp_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        dl = self._dc_lower[bar_index]
        dm = self._dc_mid[bar_index]
        stc = self._stc[bar_index]

        if np.isnan(c) or np.isnan(dl) or np.isnan(stc):
            return 0.0

        sp = self._sell_p[bar_index]
        bp = self._buy_p[bar_index]

        if c < dl and stc < 25 and not np.isnan(sp) and not np.isnan(bp) and sp > bp:
            return -0.90
        if c < dm and stc < 40:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay", "style": "dash"},
            {"name": f"DC Mid({self.dc_period})", "array": self._dc_mid, "type": "overlay"},
            {"name": f"DC Lower({self.dc_period})", "array": self._dc_lower, "type": "overlay", "style": "dash"},
            {"name": f"STC({self.schaff_period})", "array": self._stc,
             "y_range": [0, 100], "horizontal_lines": [25, 75]},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"DC Mid({self.dc_period})", datetimes, self._dc_mid, color="#ffab40"),
            self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower, style="dash", color="#26a69a"),
        ]
        subplots = [
            self._make_subplot(f"STC({self.schaff_period})", [
                self._make_subplot_trace("STC", datetimes, self._stc, color="#42a5f5"),
            ], horizontal_lines=[25, 75], y_range=[0, 100]),
        ]
        return {"overlays": overlays, "subplots": subplots}
