"""MildTrendLongI1hV7 — Keltner(30,1.8) + ROC(20) + OI_Flow(25).

Economic logic: Keltner Channel with tight multiplier captures 1H iron ore intraday
trend bands. ROC above zero confirms positive price momentum. OI flow validates that
new positions align with price direction. Signal scales with ROC magnitude and channel
position for breakout detection.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.trend.keltner import keltner
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV7(TrendingStrategy):
    name = "mild_trend_long_I_1h_v7"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 40

    kelt_period: int = 30
    kelt_mult: float = 1.8
    roc_period: int = 20
    oif_period: int = 25
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_period, atr_period=self.kelt_period // 2, multiplier=self.kelt_mult
        )
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._oi_flow, self._oi_flow_sig = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oif_period
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        km = self._kelt_mid[bar_index]
        ku = self._kelt_upper[bar_index]
        roc_val = self._roc[bar_index]
        oif = self._oi_flow[bar_index]
        oif_sig = self._oi_flow_sig[bar_index]

        if any(np.isnan(v) for v in [close, km, ku, roc_val, oif, oif_sig]):
            return 0.0

        above_mid = close > km
        roc_positive = roc_val > 0.0
        oi_flow_bullish = oif > oif_sig

        if not (above_mid and roc_positive and oi_flow_bullish):
            return 0.0

        roc_score = min(1.0, roc_val / 3.0) * 0.4
        breakout = 0.2 if close > ku else 0.0
        return min(1.0, 0.3 + roc_score + breakout)

    def get_indicator_config(self):
        return [
            {"name": "Keltner Upper", "array": self._kelt_upper, "type": "overlay", "style": "dash"},
            {"name": "Keltner Mid", "array": self._kelt_mid, "type": "overlay"},
            {"name": "Keltner Lower", "array": self._kelt_lower, "type": "overlay", "style": "dash"},
            {"name": f"ROC({self.roc_period})", "array": self._roc, "type": "subplot", "zero_line": True},
            {"name": "OI Flow", "array": self._oi_flow, "type": "subplot", "panel": "OI Flow", "zero_line": True},
            {"name": "OI Flow Sig", "array": self._oi_flow_sig, "type": "subplot", "panel": "OI Flow", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("Keltner Upper", datetimes, self._kelt_upper, style="dash", color="#78909c"),
                self._make_overlay("Keltner Mid", datetimes, self._kelt_mid, color="#ffab40"),
                self._make_overlay("Keltner Lower", datetimes, self._kelt_lower, style="dash", color="#78909c"),
            ],
            "subplots": [
                self._make_subplot(
                    "ROC",
                    [self._make_subplot_trace("ROC", datetimes, self._roc, color="#42a5f5")],
                    zero_line=True,
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
