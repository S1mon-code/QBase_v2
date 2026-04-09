"""MildTrendLongI4hV35 — Donchian(35) upper break + MFI(14) > 55.

Economic logic: Donchian channel breakout on 4h with wider lookback captures
significant new highs. MFI above 55 confirms money flow supports the breakout.
Dual confirmation for high-conviction trend entries.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV35(TrendingStrategy):
    name = "mild_trend_long_I_4h_v35"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 50

    dc_period: int = 35
    mfi_period: int = 14
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_mid, self._dc_lower = donchian(
            self._highs, self._lows, period=self.dc_period)
        self._mfi = mfi(self._highs, self._lows, self._closes, self._volumes,
                        period=self.mfi_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_up = self._dc_upper[bar_index]
        mfi_val = self._mfi[bar_index]

        if any(np.isnan(v) for v in [close, dc_up, mfi_val]):
            return 0.0

        breakout = close >= dc_up
        mfi_strong = mfi_val > 55.0

        if not (breakout and mfi_strong):
            return 0.0

        mfi_score = min(0.6, max(0.0, (mfi_val - 55.0) / 41.7)) + 0.4
        return min(1.0, mfi_score)

    def get_indicator_config(self):
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay"},
            {"name": f"DC Mid({self.dc_period})", "array": self._dc_mid, "type": "overlay",
             "style": "dash"},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [55]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("DC Upper", datetimes, self._dc_upper, color="#ff7043"),
                self._make_overlay("DC Mid", datetimes, self._dc_mid, color="#90a4ae",
                                   style="dash"),
            ],
            "subplots": [
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#26a69a")],
                    horizontal_lines=[55], y_range=[0, 100],
                ),
            ],
        }
