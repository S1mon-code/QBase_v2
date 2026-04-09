"""StrongTrendShortAG4hV3 — HMA Bearish + Fisher Transform + Wyckoff Distribution.

Economic logic: HMA responds quickly to silver's 4H moves. Fisher Transform
below -1 signals bearish extremes. Wyckoff bearish divergence detects
distribution where smart money sells into retail buying.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.fisher_transform import fisher_transform
from indicators.trend.hma import hma
from indicators.volume.wyckoff_divergence import wyckoff_divergence
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV3(TrendingStrategy):
    """HMA bearish + Fisher negative + Wyckoff distribution.

    Signal logic:
        close < HMA AND Fisher < -1 AND bear_div > 0 -> -0.85
        close < HMA AND Fisher < 0 -> -0.45
        else -> 0.0
    """

    name = "short_AG_4h_v3"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    hma_period: int = 35
    fisher_period: int = 25
    wyckoff_lookback: int = 50
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, self.hma_period)
        self._fisher, self._fisher_signal = fisher_transform(
            self._highs, self._lows, period=self.fisher_period,
        )
        _, self._bear_div, _ = wyckoff_divergence(
            self._highs, self._lows, self._closes, self._volumes,
            lookback=self.wyckoff_lookback,
        )

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        h = self._hma[bar_index]
        f = self._fisher[bar_index]
        bd = self._bear_div[bar_index]

        if np.isnan(c) or np.isnan(h) or np.isnan(f):
            return 0.0

        if c >= h:
            return 0.0

        if f < -1 and (not np.isnan(bd) and bd > 0):
            return -0.85
        if f < 0:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma, "type": "overlay"},
            {"name": "Fisher", "array": self._fisher, "panel": "Fisher", "zero_line": True},
            {"name": "Fisher Signal", "array": self._fisher_signal, "panel": "Fisher"},
            {"name": "Wyckoff Bear Div", "array": self._bear_div, "style": "bar"},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot("Fisher Transform", [
                self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#42a5f5"),
                self._make_subplot_trace("Fisher Signal", datetimes, self._fisher_signal, color="#ff7043"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
