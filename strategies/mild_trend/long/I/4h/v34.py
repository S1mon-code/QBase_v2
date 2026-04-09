"""MildTrendLongI4hV34 — Bollinger(22) squeeze + MACD(14,30,9) bullish.

Economic logic: Bollinger squeeze on 4h detects volatility compression before major
directional moves. MACD with standard-ish parameters confirms breakout direction.
Signal scales with MACD spread and bandwidth expansion.
"""
from __future__ import annotations

import numpy as np

from indicators.volatility.bollinger import bollinger_bands
from indicators.momentum.macd import macd
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV34(TrendingStrategy):
    name = "mild_trend_long_I_4h_v34"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup = 55

    bb_period: int = 22
    macd_fast: int = 14
    macd_slow: int = 30
    macd_signal: int = 9
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            self._closes, period=self.bb_period)
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow,
            signal=self.macd_signal)
        n = len(self._closes)
        self._bandwidth = np.full(n, np.nan)
        for i in range(n):
            if not np.isnan(self._bb_upper[i]) and self._bb_mid[i] != 0:
                self._bandwidth[i] = (self._bb_upper[i] - self._bb_lower[i]) / self._bb_mid[i]

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        macd_l = self._macd_line[bar_index]
        macd_s = self._macd_signal[bar_index]
        bw = self._bandwidth[bar_index]
        bw_prev = self._bandwidth[bar_index - 1] if bar_index > 0 else np.nan

        if any(np.isnan(v) for v in [close, macd_l, macd_s, bw, bw_prev]):
            return 0.0

        squeeze_release = bw > bw_prev
        price_strong = close > self._bb_mid[bar_index]
        macd_bullish = macd_l > macd_s

        if not (squeeze_release and price_strong and macd_bullish):
            return 0.0

        macd_spread = (macd_l - macd_s) / abs(macd_s) if macd_s != 0 else 0.0
        macd_score = min(0.5, max(0.0, macd_spread * 2.0)) + 0.3
        if close > self._bb_upper[bar_index]:
            macd_score += 0.15
        return min(1.0, macd_score)

    def get_indicator_config(self):
        return [
            {"name": f"BB Upper({self.bb_period})", "array": self._bb_upper, "type": "overlay"},
            {"name": f"BB Mid({self.bb_period})", "array": self._bb_mid, "type": "overlay",
             "style": "dash"},
            {"name": f"BB Lower({self.bb_period})", "array": self._bb_lower, "type": "overlay"},
            {"name": "MACD", "array": self._macd_line, "type": "subplot", "panel": "MACD"},
            {"name": "Signal", "array": self._macd_signal, "type": "subplot", "panel": "MACD",
             "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("BB Upper", datetimes, self._bb_upper, color="#42a5f5"),
                self._make_overlay("BB Mid", datetimes, self._bb_mid, color="#90a4ae",
                                   style="dash"),
                self._make_overlay("BB Lower", datetimes, self._bb_lower, color="#42a5f5"),
            ],
            "subplots": [
                self._make_subplot(
                    "MACD",
                    [
                        self._make_subplot_trace("MACD", datetimes, self._macd_line,
                                                 color="#26a69a"),
                        self._make_subplot_trace("Signal", datetimes, self._macd_signal,
                                                 color="#ef5350", style="dash"),
                    ],
                    horizontal_lines=[0],
                ),
            ],
        }
