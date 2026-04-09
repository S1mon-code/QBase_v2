"""MildTrendShortI4hV8 — TEMA Below Price + Williams %R Bearish + OI Flow Negative.

Economic logic: Price below TEMA on 4H signals triple-smoothed bearish trend.
Williams %R below -50 confirms price is in the lower half of the range.
Negative OI flow indicates position building against price direction.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.tema import tema
from indicators.momentum.williams_r import williams_r
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV8(TrendingStrategy):
    """Price below TEMA(30) + Williams%R(30)<-50 + OI_Flow(35) negative."""

    name = "short_I_4h_v8"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    tema_period: int = 30
    wr_period: int = 30
    oi_flow_period: int = 35
    chandelier_mult: float = 2.5

    def on_init_arrays(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        volumes: np.ndarray,
        oi: np.ndarray,
        datetimes: np.ndarray,
    ) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, self.tema_period)
        self._wr = williams_r(self._highs, self._lows, self._closes, self.wr_period)
        self._oi_flow, self._oi_flow_sig = oi_flow(
            self._closes, self._oi, self._volumes, self.oi_flow_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        t = self._tema[bar_index]
        wr = self._wr[bar_index]
        oif = self._oi_flow[bar_index]
        oif_sig = self._oi_flow_sig[bar_index]

        if any(np.isnan(v) for v in (close, t, wr)):
            return 0.0

        if close >= t:
            return 0.0

        dist = (t - close) / t
        strength = min(1.0, dist * 35.0)

        signal = -(0.25 + strength * 0.3)

        if wr < -50:
            wr_str = min(1.0, abs(wr + 50) / 30.0)
            signal -= 0.2 * wr_str

        if not np.isnan(oif) and not np.isnan(oif_sig) and oif < oif_sig:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay", "color": "#ffab40"},
            {"name": f"W%R({self.wr_period})", "array": self._wr, "type": "subplot",
             "y_range": [-100, 0], "horizontal_lines": [-20, -50, -80]},
            {"name": "OI Flow", "array": self._oi_flow, "type": "subplot", "panel": "OI Flow", "zero_line": True},
            {"name": "OI Flow Sig", "array": self._oi_flow_sig, "type": "subplot", "panel": "OI Flow", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"W%R({self.wr_period})",
                    [self._make_subplot_trace("W%R", datetimes, self._wr, color="#bb86fc")],
                    horizontal_lines=[-20, -50, -80], y_range=[-100, 0],
                ),
                self._make_subplot(
                    "OI Flow",
                    [
                        self._make_subplot_trace("Flow", datetimes, self._oi_flow, color="#26a69a"),
                        self._make_subplot_trace("Signal", datetimes, self._oi_flow_sig, style="dash", color="#78909c"),
                    ],
                    zero_line=True,
                ),
            ],
        }
