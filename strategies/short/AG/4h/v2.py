"""StrongTrendShortAG4hV2 — SuperTrend Bear + RSI Weak + Volume Spike.

Economic logic: SuperTrend bearish direction on 4H confirms intraday downtrend.
RSI below 45 shows momentum is bearish. Volume spike confirms panic selling or
institutional distribution backing the move.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.rsi import rsi
from indicators.trend.supertrend import supertrend
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV2(TrendingStrategy):
    """SuperTrend bear + RSI weak + volume spike confirmation.

    Signal logic:
        ST bearish AND RSI < 40 AND volume_spike > 0 -> -0.90
        ST bearish AND RSI < 45 -> -0.50
        else -> 0.0
    """

    name = "short_AG_4h_v2"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    st_period: int = 40
    st_mult: float = 3.5
    rsi_period: int = 30
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        self._rsi = rsi(self._closes, period=self.rsi_period)
        self._vspike = volume_spike(self._volumes, period=25, threshold=1.8)

    def _generate_signal(self, bar_index: int) -> float:
        sd = self._st_dir[bar_index]
        r = self._rsi[bar_index]
        vs = self._vspike[bar_index]

        if np.isnan(sd) or np.isnan(r):
            return 0.0

        if sd > 0:  # bullish
            return 0.0

        if r < 40 and (not np.isnan(vs) and vs > 0):
            return -0.90
        if r < 45:
            return -0.50
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"SuperTrend({self.st_period})", "array": self._st_line, "type": "overlay"},
            {"name": f"RSI({self.rsi_period})", "array": self._rsi,
             "y_range": [0, 100], "horizontal_lines": [40, 60]},
            {"name": "VolSpike", "array": self._vspike, "style": "bar"},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"SuperTrend({self.st_period})", datetimes, self._st_line, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot(f"RSI({self.rsi_period})", [
                self._make_subplot_trace("RSI", datetimes, self._rsi, color="#42a5f5"),
            ], horizontal_lines=[40, 60], y_range=[0, 100]),
            self._make_subplot("VolSpike", [
                self._make_subplot_trace("VolSpike", datetimes, self._vspike, style="bar", color="#ab47bc"),
            ]),
        ]
        return {"overlays": overlays, "subplots": subplots}
