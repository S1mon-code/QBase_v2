"""MildTrendLongI4hV7 — Donchian(50) + TSI(30,15) + ForceIndex(30).

Economic logic: Donchian channel breakout captures 4H iron ore price expansion.
TSI double-smoothed momentum confirms trend direction with reduced whipsaws.
Force Index combines price movement with volume for conviction measure.
Signal scales with TSI magnitude and price position within channel.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.trend.donchian import donchian
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV7(TrendingStrategy):
    name = "long_I_4h_v7"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 60

    dc_period: int = 50
    tsi_long: int = 30
    tsi_short: int = 15
    fi_period: int = 30
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, period=self.dc_period
        )
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=self.tsi_short
        )
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_up = self._dc_upper[bar_index]
        dc_mid = self._dc_mid[bar_index]
        tsi_val = self._tsi_line[bar_index]
        fi_val = self._fi[bar_index]

        if any(np.isnan(v) for v in [close, dc_up, dc_mid, tsi_val, fi_val]):
            return 0.0

        above_mid = close > dc_mid
        tsi_bullish = tsi_val > 0.0
        fi_positive = fi_val > 0.0

        if not (above_mid and tsi_bullish and fi_positive):
            return 0.0

        # Bonus for near-upper breakout
        channel_pos = (close - dc_mid) / (dc_up - dc_mid) if dc_up > dc_mid else 0.0
        channel_score = min(1.0, channel_pos) * 0.3
        tsi_score = min(1.0, tsi_val / 20.0) * 0.4
        return min(1.0, 0.3 + channel_score + tsi_score)

    def get_indicator_config(self):
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay", "style": "dash"},
            {"name": "DC Mid", "array": self._dc_mid, "type": "overlay"},
            {"name": f"DC Lower({self.dc_period})", "array": self._dc_lower, "type": "overlay", "style": "dash"},
            {"name": "TSI", "array": self._tsi_line, "type": "subplot", "panel": "TSI", "zero_line": True},
            {"name": "TSI Signal", "array": self._tsi_signal, "type": "subplot", "panel": "TSI", "style": "dash"},
            {"name": f"Force Index({self.fi_period})", "array": self._fi, "type": "subplot", "zero_line": True},
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
                    "Force Index",
                    [self._make_subplot_trace("FI", datetimes, self._fi, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
