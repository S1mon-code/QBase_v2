"""MildTrendShortI4hV20 — Fisher Transform bearish + CMF distribution.

Economic logic: Fisher Transform(14) < 0 on 4H Iron Ore normalizes price
into a Gaussian distribution, making bearish extremes more identifiable.
CMF(25) < 0 confirms institutional distribution, adding volume-based
conviction to the short signal.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.fisher_transform import fisher_transform
from indicators.trend.ema import ema
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV20(TrendingStrategy):
    """Fisher Transform(14) < 0 + CMF(25) < 0."""

    name = "short_I_4h_v20"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 40

    fisher_period: int = 14
    cmf_period: int = 25
    chandelier_mult: float = 3.4

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
        self._fisher, self._fisher_sig = fisher_transform(
            self._highs, self._lows, self.fisher_period,
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, bar_index: int) -> float:
        f = self._fisher[bar_index]
        fs = self._fisher_sig[bar_index]
        c = self._cmf[bar_index]

        if any(np.isnan(v) for v in (f, fs, c)):
            return 0.0

        # Fisher must be bearish (negative and below signal)
        if f >= 0:
            return 0.0

        signal = -(0.3 + min(0.2, abs(f) * 0.15))

        # Extra if Fisher below signal
        if f < fs:
            signal -= 0.1

        # CMF distribution confirmation
        if c < 0:
            signal -= min(0.1, abs(c) * 0.4)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"Fisher({self.fisher_period})", "array": self._fisher, "type": "subplot", "zero_line": True},
            {"name": "Fisher Signal", "array": self._fisher_sig, "type": "subplot",
             "panel": f"Fisher({self.fisher_period})", "color": "#ff9800"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"Fisher({self.fisher_period})",
                    [
                        self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#bb86fc"),
                        self._make_subplot_trace("Signal", datetimes, self._fisher_sig, color="#ff9800"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26c6da")],
                    zero_line=True,
                ),
            ],
        }
