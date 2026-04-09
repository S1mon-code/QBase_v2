"""MildTrendLongIDailyV6 — TEMA(80) + Schaff(100,60,30) + OI_Flow(100).

Economic logic: TEMA's triple-smoothing reduces lag while filtering daily noise on
iron ore. Schaff Trend Cycle combines MACD with double stochastic smoothing — readings
above 25 confirm early trend entry. OI flow shows whether new positions align with
price direction. Three orthogonal signals provide robust trend confirmation.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.trend.tema import tema
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV6(TrendingStrategy):
    name = "mild_trend_long_I_daily_v6"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 130

    tema_period: int = 80
    schaff_period: int = 100
    oi_flow_period: int = 100
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, period=self.tema_period)
        self._schaff = schaff_trend_cycle(self._closes, period=self.schaff_period, fast=60, slow=30)
        self._oi_flow, self._oi_flow_sig = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oi_flow_period
        )

    def _generate_signal(self, bar_index: int) -> float:
        t = self._tema[bar_index]
        t_prev = self._tema[bar_index - 1] if bar_index > 0 else np.nan
        stc = self._schaff[bar_index]
        oif = self._oi_flow[bar_index]
        oif_sig = self._oi_flow_sig[bar_index]

        if any(np.isnan(v) for v in [t, t_prev, stc, oif, oif_sig]):
            return 0.0

        tema_rising = t > t_prev
        schaff_bullish = stc > 25.0
        oi_flow_positive = oif > oif_sig

        if not (tema_rising and schaff_bullish and oi_flow_positive):
            return 0.0

        schaff_score = min(1.0, (stc - 25.0) / 75.0) * 0.5
        base = 0.3
        return min(1.0, base + schaff_score + 0.2)

    def get_indicator_config(self):
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay"},
            {"name": f"Schaff({self.schaff_period})", "array": self._schaff, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [25, 75]},
            {"name": "OI Flow", "array": self._oi_flow, "type": "subplot", "panel": "OI Flow", "zero_line": True},
            {"name": "OI Flow Sig", "array": self._oi_flow_sig, "type": "subplot", "panel": "OI Flow", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Schaff Trend Cycle",
                    [self._make_subplot_trace("STC", datetimes, self._schaff, color="#bb86fc")],
                    y_range=[0, 100], horizontal_lines=[25, 75],
                ),
                self._make_subplot(
                    "OI Flow",
                    [
                        self._make_subplot_trace("Flow", datetimes, self._oi_flow, color="#66bb6a"),
                        self._make_subplot_trace("Signal", datetimes, self._oi_flow_sig, color="#ef5350", style="dash"),
                    ],
                    zero_line=True,
                ),
            ],
        }
