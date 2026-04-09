"""MildTrendShortI4hV10 — PSAR Bearish + Aroon Bearish + Twiggs Negative.

Economic logic: PSAR above price confirms bearish trend state on 4H.
Aroon Down exceeding Aroon Up signals the asset is making new lows more
frequently. Twiggs Money Flow below zero validates persistent selling.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.psar import psar
from indicators.trend.aroon import aroon
from indicators.volume.twiggs import twiggs_money_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV10(TrendingStrategy):
    """PSAR bearish + Aroon(35) bearish + Twiggs(30)<0."""

    name = "short_I_4h_v10"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    aroon_period: int = 35
    twiggs_period: int = 30
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
        self._psar_vals, self._psar_dir = psar(self._highs, self._lows)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, self.aroon_period,
        )
        self._twiggs = twiggs_money_flow(
            self._highs, self._lows, self._closes, self._volumes, self.twiggs_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        psar_d = self._psar_dir[bar_index]
        ar_up = self._aroon_up[bar_index]
        ar_dn = self._aroon_down[bar_index]
        tw = self._twiggs[bar_index]

        if any(np.isnan(v) for v in (psar_d, ar_up, ar_dn)):
            return 0.0

        if psar_d > 0:
            return 0.0

        signal = -0.3

        if ar_dn > ar_up:
            aroon_str = min(1.0, (ar_dn - ar_up) / 60.0)
            signal -= 0.25 * aroon_str

        if not np.isnan(tw) and tw < 0:
            signal -= 0.2

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "PSAR", "array": self._psar_vals, "type": "overlay", "style": "step", "color": "#ef5350"},
            {"name": "Aroon Up", "array": self._aroon_up, "type": "subplot", "panel": "Aroon"},
            {"name": "Aroon Down", "array": self._aroon_down, "type": "subplot", "panel": "Aroon"},
            {"name": f"Twiggs({self.twiggs_period})", "array": self._twiggs, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay("PSAR", datetimes, self._psar_vals, style="step", color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Aroon({self.aroon_period})",
                    [
                        self._make_subplot_trace("Up", datetimes, self._aroon_up, color="#26a69a"),
                        self._make_subplot_trace("Down", datetimes, self._aroon_down, color="#ef5350"),
                    ],
                    y_range=[0, 100],
                ),
                self._make_subplot(
                    f"Twiggs({self.twiggs_period})",
                    [self._make_subplot_trace("Twiggs", datetimes, self._twiggs, color="#bb86fc")],
                    zero_line=True,
                ),
            ],
        }
