"""StrongTrendShortAG4hV7 — Donchian Breakdown + TSI Negative + BSP Selling.

Economic logic: Price breaking Donchian lower on 4H makes new swing lows.
TSI negative validates momentum is bearish. Selling pressure exceeding buying
confirms distribution in silver.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.trend.donchian import donchian
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV7(TrendingStrategy):
    """Donchian breakdown + TSI negative + selling pressure dominant.

    Signal logic:
        close < DC_lower AND TSI < -5 AND sell > buy -> -0.85
        close < DC_mid AND TSI < 0 -> -0.45
        else -> 0.0
    """

    name = "short_AG_4h_v7"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    dc_period: int = 45
    tsi_long: int = 25
    bsp_period: int = 25
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.dc_period,
        )
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=12,
        )
        self._buy_p, self._sell_p, _ = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.bsp_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        dl = self._dc_lower[bar_index]
        dm = self._dc_mid[bar_index]
        t = self._tsi_line[bar_index]

        if np.isnan(c) or np.isnan(dl) or np.isnan(t):
            return 0.0

        sp = self._sell_p[bar_index]
        bp = self._buy_p[bar_index]

        if c < dl and t < -5 and not np.isnan(sp) and not np.isnan(bp) and sp > bp:
            return -0.85
        if c < dm and t < 0:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay", "style": "dash"},
            {"name": f"DC Mid({self.dc_period})", "array": self._dc_mid, "type": "overlay"},
            {"name": f"DC Lower({self.dc_period})", "array": self._dc_lower, "type": "overlay", "style": "dash"},
            {"name": "TSI", "array": self._tsi_line, "panel": "TSI", "zero_line": True},
            {"name": "TSI Signal", "array": self._tsi_signal, "panel": "TSI"},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"DC Mid({self.dc_period})", datetimes, self._dc_mid, color="#ffab40"),
            self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower, style="dash", color="#26a69a"),
        ]
        subplots = [
            self._make_subplot("TSI", [
                self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#42a5f5"),
                self._make_subplot_trace("TSI Signal", datetimes, self._tsi_signal, color="#ff7043"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
