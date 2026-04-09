"""MildTrendLongI1hV9 — Donchian(30) + TSI(20,10) + OBV(30).

Economic logic: Donchian channel breakout captures 1H iron ore intraday price
expansion. TSI double-smoothed momentum confirms trend direction. Rising OBV
validates volume participation. Signal scales with TSI magnitude and channel
position for gradual entry sizing.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.trend.donchian import donchian
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV9(TrendingStrategy):
    name = "mild_trend_long_I_1h_v9"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 40

    dc_period: int = 30
    tsi_long: int = 20
    tsi_short: int = 10
    obv_smooth: int = 30
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.dc_period
        )
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=self.tsi_short
        )
        self._obv_raw = obv(self._closes, self._volumes)
        n = len(self._closes)
        self._obv_ma = np.full(n, np.nan)
        for i in range(self.obv_smooth - 1, n):
            self._obv_ma[i] = np.mean(self._obv_raw[i - self.obv_smooth + 1:i + 1])

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_up = self._dc_upper[bar_index]
        dc_mid = self._dc_mid[bar_index]
        tsi_val = self._tsi_line[bar_index]
        obv_now = self._obv_raw[bar_index]
        obv_ma = self._obv_ma[bar_index]

        if any(np.isnan(v) for v in [close, dc_up, dc_mid, tsi_val, obv_now, obv_ma]):
            return 0.0

        above_mid = close > dc_mid
        tsi_bullish = tsi_val > 0.0
        obv_rising = obv_now > obv_ma

        if not (above_mid and tsi_bullish and obv_rising):
            return 0.0

        tsi_score = min(1.0, tsi_val / 15.0) * 0.4
        channel_score = min(1.0, (close - dc_mid) / (dc_up - dc_mid)) * 0.3 if dc_up > dc_mid else 0.0
        return min(1.0, 0.3 + tsi_score + channel_score)

    def get_indicator_config(self):
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay", "style": "dash"},
            {"name": "DC Mid", "array": self._dc_mid, "type": "overlay"},
            {"name": f"DC Lower({self.dc_period})", "array": self._dc_lower, "type": "overlay", "style": "dash"},
            {"name": "TSI", "array": self._tsi_line, "type": "subplot", "panel": "TSI", "zero_line": True},
            {"name": "TSI Signal", "array": self._tsi_signal, "type": "subplot", "panel": "TSI", "style": "dash"},
            {"name": "OBV", "array": self._obv_raw, "type": "subplot", "panel": "OBV"},
            {"name": "OBV MA", "array": self._obv_ma, "type": "subplot", "panel": "OBV", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, style="dash", color="#78909c"),
                self._make_overlay("DC Mid", datetimes, self._dc_mid, color="#ffab40"),
                self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower, style="dash", color="#78909c"),
            ],
            "subplots": [
                self._make_subplot(
                    "TSI",
                    [
                        self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#bb86fc"),
                        self._make_subplot_trace("Signal", datetimes, self._tsi_signal, color="#ff8a80", style="dash"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv_raw, color="#66bb6a"),
                        self._make_subplot_trace("OBV MA", datetimes, self._obv_ma, color="#ef5350", style="dash"),
                    ],
                ),
            ],
        }
