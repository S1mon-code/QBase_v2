"""MildTrendLongI2hV8 — Donchian(60) + AwesomeOsc(40) + Twiggs(50).

Economic logic: Donchian channel breakout captures 2H iron ore price expansion.
Awesome Oscillator (midpoint SMA differential) confirms momentum direction.
Twiggs Money Flow detects institutional accumulation using true-range-adjusted
volume. Signal scales with channel position and TMF level.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.awesome_oscillator import ao
from indicators.trend.donchian import donchian
from indicators.volume.twiggs import twiggs_money_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV8(TrendingStrategy):
    name = "long_I_2h_v8"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 70

    dc_period: int = 60
    ao_slow: int = 40
    tmf_period: int = 50
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.dc_period
        )
        self._ao = ao(self._highs, self._lows, fast=5, slow=self.ao_slow)
        self._tmf = twiggs_money_flow(
            self._highs, self._lows, self._closes, self._volumes, period=self.tmf_period
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_up = self._dc_upper[bar_index]
        dc_mid = self._dc_mid[bar_index]
        ao_val = self._ao[bar_index]
        tmf = self._tmf[bar_index]

        if any(np.isnan(v) for v in [close, dc_up, dc_mid, ao_val, tmf]):
            return 0.0

        above_mid = close > dc_mid
        ao_positive = ao_val > 0.0
        tmf_positive = tmf > 0.0

        if not (above_mid and ao_positive and tmf_positive):
            return 0.0

        # Channel position score
        channel_score = min(1.0, (close - dc_mid) / (dc_up - dc_mid)) * 0.3 if dc_up > dc_mid else 0.0
        tmf_score = min(1.0, tmf / 0.2) * 0.3
        return min(1.0, 0.3 + channel_score + tmf_score)

    def get_indicator_config(self):
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay", "style": "dash"},
            {"name": "DC Mid", "array": self._dc_mid, "type": "overlay"},
            {"name": f"DC Lower({self.dc_period})", "array": self._dc_lower, "type": "overlay", "style": "dash"},
            {"name": "AO", "array": self._ao, "type": "subplot", "style": "bar",
             "color_positive": "#66bb6a", "color_negative": "#ef5350", "zero_line": True},
            {"name": f"TMF({self.tmf_period})", "array": self._tmf, "type": "subplot", "zero_line": True},
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
                    "Awesome Oscillator",
                    [self._make_subplot_trace("AO", datetimes, self._ao, style="bar",
                                              color_positive="#66bb6a", color_negative="#ef5350")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "Twiggs MF",
                    [self._make_subplot_trace("TMF", datetimes, self._tmf, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
