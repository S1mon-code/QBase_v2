"""StrongTrendLongAG2hV9 — Donchian(60) + MACD(14,38,10) + MFI(50).

Economic logic: Donchian breakout on 2H captures AG's range expansion.
MACD (substituted for AwesomeOsc for cleaner signal) provides momentum
direction. MFI acts as volume-weighted RSI for money flow confirmation.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.momentum.macd import macd
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG2hV9(TrendingStrategy):
    name = "long_AG_2h_v9"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    donchian_period: int = 60
    macd_fast: int = 14
    macd_slow: int = 38
    mfi_period: int = 50
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.donchian_period,
        )
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow, signal=10,
        )
        self._mfi = mfi(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.mfi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_mid = self._dc_mid[bar_index]
        ml = self._macd_line[bar_index]
        m = self._mfi[bar_index]

        if any(np.isnan(v) for v in (close, dc_mid, ml, m)):
            return 0.0

        if close > dc_mid and ml > 0.0 and m > 50.0:
            strength = min(1.0, (m - 50.0) / 30.0 * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"DC Upper({self.donchian_period})", "array": self._dc_upper, "type": "overlay"},
            {"name": f"DC Mid({self.donchian_period})", "array": self._dc_mid, "type": "overlay"},
            {"name": f"DC Lower({self.donchian_period})", "array": self._dc_lower, "type": "overlay"},
            {"name": f"MACD({self.macd_fast},{self.macd_slow})", "array": self._macd_line,
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
                    f"MACD({self.macd_fast},{self.macd_slow})",
                    [self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._macd_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#42a5f5")],
                    y_range=[0, 100], horizontal_lines=[20, 50, 80],
                ),
            ],
        }
