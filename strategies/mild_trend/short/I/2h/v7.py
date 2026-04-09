"""MildTrendShortI2hV7 — ZLEMA Below Price + Fisher Bearish + OI Flow Negative.

Economic logic: ZLEMA's zero-lag property captures trend changes quickly on 2H.
Fisher Transform crossing below trigger signals bearish momentum. Negative
OI flow confirms positions are built against rising price.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.zlema import zlema
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV7(TrendingStrategy):
    """Price below ZLEMA(35) + Fisher(25) bearish + OI_Flow(50) negative."""

    name = "mild_trend_short_I_2h_v7"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    zlema_period: int = 35
    fisher_period: int = 25
    oi_flow_period: int = 50
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
        self._zlema = zlema(self._closes, self.zlema_period)
        self._fisher, self._fisher_trigger = fisher_transform(
            self._highs, self._lows, self.fisher_period,
        )
        self._oi_flow, self._oi_flow_sig = oi_flow(
            self._closes, self._oi, self._volumes, self.oi_flow_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        zl = self._zlema[bar_index]
        fish = self._fisher[bar_index]
        fish_trig = self._fisher_trigger[bar_index]
        oif = self._oi_flow[bar_index]
        oif_sig = self._oi_flow_sig[bar_index]

        if any(np.isnan(v) for v in (close, zl, fish, fish_trig)):
            return 0.0

        if close >= zl:
            return 0.0

        dist = (zl - close) / zl
        strength = min(1.0, dist * 35.0)

        signal = -(0.25 + strength * 0.3)

        if fish < fish_trig:
            signal -= 0.2

        if not np.isnan(oif) and not np.isnan(oif_sig) and oif < oif_sig:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"ZLEMA({self.zlema_period})", "array": self._zlema, "type": "overlay", "color": "#ffab40"},
            {"name": "Fisher", "array": self._fisher, "type": "subplot", "panel": "Fisher", "zero_line": True},
            {"name": "Trigger", "array": self._fisher_trigger, "type": "subplot", "panel": "Fisher", "style": "dash"},
            {"name": "OI Flow", "array": self._oi_flow, "type": "subplot", "panel": "OI Flow", "zero_line": True},
            {"name": "OI Flow Sig", "array": self._oi_flow_sig, "type": "subplot", "panel": "OI Flow", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"ZLEMA({self.zlema_period})", datetimes, self._zlema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Fisher",
                    [
                        self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#bb86fc"),
                        self._make_subplot_trace("Trigger", datetimes, self._fisher_trigger, style="dash", color="#78909c"),
                    ],
                    zero_line=True,
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
