"""MildTrendShortI1hV6 — KAMA slope + Fisher bearish + CMF distribution.

Economic logic: KAMA(16) declining slope on 1H adapts to market efficiency and
confirms bearish trend. Fisher Transform(12) < -0.4 signals statistically
significant downward momentum. CMF(12) < 0 validates institutional distribution.
Signal smoothed with EMA(3) for mild-trend regime stability.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.cmf import cmf
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV6(TrendingStrategy):
    """KAMA(16) slope < 0 + Fisher(12) < -0.4 + CMF(12) < 0."""

    name = "mild_trend_short_I_1h_v6"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    kama_period: int = 16
    fisher_period: int = 12
    cmf_period: int = 12
    chandelier_mult: float = 3.0

    _kama_line: np.ndarray | None = None
    _fisher_line: np.ndarray | None = None
    _fisher_trigger: np.ndarray | None = None
    _cmf_arr: np.ndarray | None = None
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
        self._kama_line = kama(self._closes, self.kama_period)
        self._fisher_line, self._fisher_trigger = fisher_transform(
            self._highs, self._lows, self.fisher_period,
        )
        self._cmf_arr = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        k = self._kama_line[bar_index]
        k_prev = self._kama_line[bar_index - 1]
        f = self._fisher_line[bar_index]
        c = self._cmf_arr[bar_index]

        if any(np.isnan(v) for v in (k, k_prev, f, c)):
            return 0.0

        slope = k - k_prev
        if slope >= 0:
            return 0.0

        signal = -0.3

        if f < -0.4:
            signal -= min(0.2, abs(f + 0.4) * 0.2)

        if c < 0:
            signal -= min(0.15, abs(c) * 0.5)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama_line, "type": "overlay", "color": "#26c6da"},
            {"name": f"Fisher({self.fisher_period})", "array": self._fisher_line, "type": "subplot",
             "panel": "Fisher", "zero_line": True, "horizontal_lines": [-0.4]},
            {"name": "Fisher Trigger", "array": self._fisher_trigger, "type": "subplot",
             "panel": "Fisher", "style": "dash", "color": "#ffab40"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf_arr, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama_line, color="#26c6da"),
            ],
            "subplots": [
                self._make_subplot(
                    "Fisher",
                    [
                        self._make_subplot_trace("Fisher", datetimes, self._fisher_line, color="#bb86fc"),
                        self._make_subplot_trace("Trigger", datetimes, self._fisher_trigger,
                                                 style="dash", color="#ffab40"),
                    ],
                    zero_line=True, horizontal_lines=[-0.4],
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf_arr, color="#ef5350")],
                    zero_line=True,
                ),
            ],
        }
