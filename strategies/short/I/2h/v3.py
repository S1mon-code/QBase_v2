"""MildTrendShortI2hV3 — HMA Declining + Schaff Falling + CMF Distribution.

Economic logic: HMA declining on 2H captures low-lag bearish trend direction.
Schaff Trend Cycle falling below 75 confirms momentum exhaustion. CMF below
zero signals distribution and money flowing out.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV3(TrendingStrategy):
    """HMA(40) declining + Schaff(50,30,50) falling + CMF(45)<0."""

    name = "short_I_2h_v3"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 65

    hma_period: int = 40
    schaff_period: int = 50
    schaff_fast: int = 30
    cmf_period: int = 45
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
        self._hma = hma(self._closes, self.hma_period)
        self._schaff = schaff_trend_cycle(
            self._closes, period=self.schaff_period, fast=self.schaff_fast, slow=self.schaff_period,
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes, self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < 2:
            return 0.0

        hma_cur = self._hma[bar_index]
        hma_prev = self._hma[bar_index - 1]
        schaff_cur = self._schaff[bar_index]
        schaff_prev = self._schaff[bar_index - 1]
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in (hma_cur, hma_prev, schaff_cur, schaff_prev)):
            return 0.0

        if hma_cur >= hma_prev:
            return 0.0

        slope = (hma_prev - hma_cur) / hma_prev
        strength = min(1.0, slope * 50.0)

        signal = -(0.25 + strength * 0.25)

        if schaff_cur < schaff_prev and schaff_cur < 75:
            signal -= 0.2

        if not np.isnan(cmf_val) and cmf_val < 0:
            signal -= 0.2

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay", "color": "#ffab40"},
            {"name": "Schaff", "array": self._schaff, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [25, 75]},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Schaff",
                    [self._make_subplot_trace("Schaff", datetimes, self._schaff, color="#bb86fc")],
                    horizontal_lines=[25, 75], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
