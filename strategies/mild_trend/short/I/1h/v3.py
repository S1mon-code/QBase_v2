"""MildTrendShortI1hV3 — TEMA slope + Aroon Down dominant + Force Index negative.

Economic logic: TEMA(15) declining slope on 1H captures fast triple-smoothed
downtrend with minimal lag. Aroon Down(16) > 55 confirms recent lows are nearby.
Force Index(8) < 0 validates volume-weighted selling pressure. Signal smoothed
with EMA(3) to prevent whipsaws in mild downtrend conditions.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.tema import tema
from indicators.trend.aroon import aroon
from indicators.volume.force_index import force_index
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV3(TrendingStrategy):
    """TEMA(15) slope < 0 + Aroon Down(16) > 55 + Force Index(8) < 0."""

    name = "mild_trend_short_I_1h_v3"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    tema_period: int = 15
    aroon_period: int = 16
    fi_period: int = 8
    chandelier_mult: float = 2.9

    _tema_line: np.ndarray | None = None
    _aroon_down: np.ndarray | None = None
    _fi_arr: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

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
        self._tema_line = tema(self._closes, self.tema_period)
        _, self._aroon_down, _ = aroon(self._highs, self._lows, self.aroon_period)
        self._fi_arr = force_index(self._closes, self._volumes, self.fi_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        t = self._tema_line[bar_index]
        t_prev = self._tema_line[bar_index - 1]
        ad = self._aroon_down[bar_index]
        fi = self._fi_arr[bar_index]

        if any(np.isnan(v) for v in (t, t_prev, ad, fi)):
            return 0.0

        slope = t - t_prev
        if slope >= 0:
            return 0.0

        signal = -0.3

        if ad > 55:
            signal -= min(0.2, (ad - 55) / 200.0)

        if fi < 0:
            signal -= 0.15

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema_line, "type": "overlay", "color": "#ab47bc"},
            {"name": f"Aroon Down({self.aroon_period})", "array": self._aroon_down, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [55]},
            {"name": f"Force Index({self.fi_period})", "array": self._fi_arr, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema_line, color="#ab47bc"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Aroon Down({self.aroon_period})",
                    [self._make_subplot_trace("Aroon Down", datetimes, self._aroon_down, color="#ef5350")],
                    horizontal_lines=[55], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"Force Index({self.fi_period})",
                    [self._make_subplot_trace("Force Index", datetimes, self._fi_arr, color="#ff7043")],
                    zero_line=True,
                ),
            ],
        }
