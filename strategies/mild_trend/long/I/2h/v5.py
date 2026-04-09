"""MildTrendLongI2hV5 — FRAMA(60) + Vortex(50) + OI_Flow(60).

Economic logic: FRAMA adapts smoothing to 2H iron ore's fractal dimension. Vortex
VI+ > VI- confirms bullish trend movement. OI flow validates new positions are being
built in the price direction. Signal scales with vortex differential and OI flow
momentum for gradual entry.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.frama import frama
from indicators.trend.vortex import vortex
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV5(TrendingStrategy):
    name = "mild_trend_long_I_2h_v5"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 70

    frama_period: int = 60
    vortex_period: int = 50
    oif_period: int = 60
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._frama = frama(self._closes, period=self.frama_period)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period
        )
        self._oi_flow, self._oi_flow_sig = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oif_period
        )

    def _generate_signal(self, bar_index: int) -> float:
        fr = self._frama[bar_index]
        vip = self._vi_plus[bar_index]
        vim = self._vi_minus[bar_index]
        oif = self._oi_flow[bar_index]
        oif_sig = self._oi_flow_sig[bar_index]
        close = self._closes[bar_index]

        if any(np.isnan(v) for v in [fr, vip, vim, oif, oif_sig, close]):
            return 0.0

        above_frama = close > fr
        vortex_bullish = vip > vim
        oi_flow_positive = oif > oif_sig

        if not (above_frama and vortex_bullish and oi_flow_positive):
            return 0.0

        vortex_score = min(1.0, (vip - vim) / 0.3) * 0.4
        return min(1.0, 0.3 + vortex_score + 0.2)

    def get_indicator_config(self):
        return [
            {"name": f"FRAMA({self.frama_period})", "array": self._frama, "type": "overlay"},
            {"name": "VI+", "array": self._vi_plus, "type": "subplot", "panel": "Vortex"},
            {"name": "VI-", "array": self._vi_minus, "type": "subplot", "panel": "Vortex"},
            {"name": "OI Flow", "array": self._oi_flow, "type": "subplot", "panel": "OI Flow", "zero_line": True},
            {"name": "OI Flow Sig", "array": self._oi_flow_sig, "type": "subplot", "panel": "OI Flow", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"FRAMA({self.frama_period})", datetimes, self._frama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Vortex",
                    [
                        self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#66bb6a"),
                        self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
                    ],
                    horizontal_lines=[1.0],
                ),
                self._make_subplot(
                    "OI Flow",
                    [
                        self._make_subplot_trace("Flow", datetimes, self._oi_flow, color="#42a5f5"),
                        self._make_subplot_trace("Signal", datetimes, self._oi_flow_sig, color="#ff8a80", style="dash"),
                    ],
                    zero_line=True,
                ),
            ],
        }
