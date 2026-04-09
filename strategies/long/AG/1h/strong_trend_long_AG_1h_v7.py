"""StrongTrendLongAG1hV7 — Donchian(30) + ROC(18) + MFI(25).

Economic logic: Donchian breakout captures Silver's 1H range expansion.
Rate of Change provides raw momentum magnitude. MFI as volume-weighted
RSI confirms money flow participation in AG's fast moves.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.momentum.roc import rate_of_change
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG1hV7(TrendingStrategy):
    name = "long_AG_1h_v7"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    donchian_period: int = 30
    roc_period: int = 18
    mfi_period: int = 25
    chandelier_mult: float = 2.8

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.donchian_period,
        )
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._mfi = mfi(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.mfi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_mid = self._dc_mid[bar_index]
        roc = self._roc[bar_index]
        m = self._mfi[bar_index]

        if any(np.isnan(v) for v in (close, dc_mid, roc, m)):
            return 0.0

        if close > dc_mid and roc > 0.0 and m > 50.0:
            strength = min(1.0, roc / 3.0 * 0.4 + (m - 50.0) / 30.0 * 0.4 + 0.2)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"DC Upper({self.donchian_period})", "array": self._dc_upper, "type": "overlay"},
            {"name": f"DC Mid({self.donchian_period})", "array": self._dc_mid, "type": "overlay"},
            {"name": f"DC Lower({self.donchian_period})", "array": self._dc_lower, "type": "overlay"},
            {"name": f"ROC({self.roc_period})", "array": self._roc,
             "type": "subplot", "zero_line": True},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi,
             "type": "subplot", "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("DC Upper", datetimes, self._dc_upper, color="#66bb6a"),
                self._make_overlay("DC Mid", datetimes, self._dc_mid, style="dash", color="#ffab40"),
                self._make_overlay("DC Lower", datetimes, self._dc_lower, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    f"ROC({self.roc_period})",
                    [self._make_subplot_trace("ROC", datetimes, self._roc, color="#ab47bc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#42a5f5")],
                    y_range=[0, 100], horizontal_lines=[20, 50, 80],
                ),
            ],
        }
