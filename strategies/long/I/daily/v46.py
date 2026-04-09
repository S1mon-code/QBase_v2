"""MildTrendLongIDailyV46 — Donchian(40) upper break + MFI(18) > 55 + ADX(22) > 22.

Economic logic: Donchian channel breakout captures new highs — a classic trend-following
signal. MFI confirms money flowing into the asset, while ADX validates directional
strength. Triple confirmation reduces false breakouts common in iron ore.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.volume.mfi import mfi
from indicators.trend.adx import adx
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV46(TrendingStrategy):
    name = "long_I_daily_v46"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 80

    dc_period: int = 40
    mfi_period: int = 18
    adx_period: int = 22
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_mid, self._dc_lower = donchian(
            self._highs, self._lows, period=self.dc_period)
        self._mfi = mfi(self._highs, self._lows, self._closes, self._volumes,
                        period=self.mfi_period)
        self._adx = adx(self._highs, self._lows, self._closes, period=self.adx_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_up = self._dc_upper[bar_index]
        mfi_val = self._mfi[bar_index]
        adx_val = self._adx[bar_index]

        if any(np.isnan(v) for v in [close, dc_up, mfi_val, adx_val]):
            return 0.0

        breakout = close >= dc_up
        mfi_strong = mfi_val > 55.0
        adx_strong = adx_val > 22.0

        if not (breakout and mfi_strong and adx_strong):
            return 0.0

        # Scale by ADX strength
        adx_score = min(0.5, max(0.0, (adx_val - 22.0) / 30.0)) + 0.3
        # Boost with MFI
        mfi_boost = min(0.2, (mfi_val - 55.0) / 200.0)
        return min(1.0, adx_score + mfi_boost)

    def get_indicator_config(self):
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay"},
            {"name": f"DC Mid({self.dc_period})", "array": self._dc_mid, "type": "overlay",
             "style": "dash"},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [55]},
            {"name": f"ADX({self.adx_period})", "array": self._adx, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [22]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"DC Upper", datetimes, self._dc_upper, color="#ff7043"),
                self._make_overlay(f"DC Mid", datetimes, self._dc_mid, color="#90a4ae",
                                   style="dash"),
            ],
            "subplots": [
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#26a69a")],
                    horizontal_lines=[55], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"ADX({self.adx_period})",
                    [self._make_subplot_trace("ADX", datetimes, self._adx, color="#42a5f5")],
                    horizontal_lines=[22], y_range=[0, 100],
                ),
            ],
        }
