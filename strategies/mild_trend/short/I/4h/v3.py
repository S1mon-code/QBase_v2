"""MildTrendShortI4hV3 — HMA Declining + Fisher Bearish + Force Index Negative.

Economic logic: HMA captures trend direction changes with minimal lag on 4H.
Fisher Transform crossing below trigger signals bearish momentum shift.
Negative Force Index validates volume-weighted selling pressure.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV3(TrendingStrategy):
    """HMA(35) declining + Fisher(25) bearish + ForceIndex(25)<0."""

    name = "mild_trend_short_I_4h_v3"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    hma_period: int = 35
    fisher_period: int = 25
    fi_period: int = 25
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
        self._fisher, self._fisher_trigger = fisher_transform(
            self._highs, self._lows, self.fisher_period,
        )
        self._fi = force_index(self._closes, self._volumes, self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < 2:
            return 0.0

        hma_cur = self._hma[bar_index]
        hma_prev = self._hma[bar_index - 1]
        fish = self._fisher[bar_index]
        fish_trig = self._fisher_trigger[bar_index]
        fi_val = self._fi[bar_index]

        if any(np.isnan(v) for v in (hma_cur, hma_prev, fish, fish_trig)):
            return 0.0

        if hma_cur >= hma_prev:
            return 0.0

        slope = (hma_prev - hma_cur) / hma_prev
        strength = min(1.0, slope * 60.0)

        signal = -(0.25 + strength * 0.25)

        if fish < fish_trig:
            signal -= 0.2

        if not np.isnan(fi_val) and fi_val < 0:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay", "color": "#ffab40"},
            {"name": "Fisher", "array": self._fisher, "type": "subplot", "panel": "Fisher", "zero_line": True},
            {"name": "Fisher Trigger", "array": self._fisher_trigger, "type": "subplot", "panel": "Fisher", "style": "dash"},
            {"name": f"ForceIndex({self.fi_period})", "array": self._fi, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ffab40"),
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
                    f"ForceIndex({self.fi_period})",
                    [self._make_subplot_trace("FI", datetimes, self._fi, color="#ef5350")],
                    zero_line=True,
                ),
            ],
        }
