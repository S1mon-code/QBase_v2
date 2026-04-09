"""MildTrendLongI4hV10 — PSAR + ROC(30) + OI_Flow(40).

Economic logic: Parabolic SAR provides adaptive trailing stops for 4H iron ore trends.
Rate of Change above zero confirms positive momentum. OI Flow shows whether new
positions are being built in the direction of price. Signal scales with ROC magnitude
and OI flow strength.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.trend.psar import psar
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV10(TrendingStrategy):
    name = "long_I_4h_v10"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 50

    roc_period: int = 30
    oif_period: int = 40
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._psar_vals, self._psar_dir = psar(self._highs, self._lows)
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._oi_flow, self._oi_flow_sig = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oif_period
        )

    def _generate_signal(self, bar_index: int) -> float:
        pd = self._psar_dir[bar_index]
        roc_val = self._roc[bar_index]
        oif = self._oi_flow[bar_index]
        oif_sig = self._oi_flow_sig[bar_index]

        if any(np.isnan(v) for v in [pd, roc_val, oif, oif_sig]):
            return 0.0

        psar_bullish = pd == 1.0
        roc_positive = roc_val > 0.0
        oi_flow_bullish = oif > oif_sig

        if not (psar_bullish and roc_positive and oi_flow_bullish):
            return 0.0

        roc_score = min(1.0, roc_val / 5.0) * 0.4
        return min(1.0, 0.3 + roc_score + 0.2)

    def get_indicator_config(self):
        return [
            {"name": "PSAR", "array": self._psar_vals, "type": "overlay", "style": "step"},
            {"name": f"ROC({self.roc_period})", "array": self._roc, "type": "subplot", "zero_line": True},
            {"name": "OI Flow", "array": self._oi_flow, "type": "subplot", "panel": "OI Flow", "zero_line": True},
            {"name": "OI Flow Sig", "array": self._oi_flow_sig, "type": "subplot", "panel": "OI Flow", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("PSAR", datetimes, self._psar_vals, style="step", color="#ff7043"),
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
