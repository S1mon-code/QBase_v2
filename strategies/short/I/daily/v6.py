"""MildTrendShortIDailyV6 — HMA Slope + Fisher Transform + CMF.

Economic logic: HMA(40) slope turning negative provides a low-lag trend
direction signal. Fisher Transform below -0.5 amplifies the bearish
price signal via Gaussian normalization. Negative CMF confirms institutional
distribution flow in the mild downtrend.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.cmf import cmf
from indicators.trend.linear_regression import linear_regression_slope
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV6(TrendingStrategy):
    """HMA(40) slope < 0 + Fisher(20) < -0.5 + CMF(20) < 0."""

    name = "short_I_daily_v6"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 65

    # Optimizable parameters (<=5 including chandelier_mult)
    hma_period: int = 40
    fisher_period: int = 20
    cmf_period: int = 20
    fisher_threshold: float = -0.5
    chandelier_mult: float = 3.5

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
        self._hma_line = hma(self._closes, self.hma_period)
        self._hma_slope = linear_regression_slope(self._hma_line, 10)
        self._fisher, self._fisher_trigger = fisher_transform(
            self._highs, self._lows, self.fisher_period,
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        # Pre-smooth raw signal
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        slope = self._hma_slope[bar_index]
        fisher_val = self._fisher[bar_index]
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in (slope, fisher_val, cmf_val)):
            return 0.0

        # HMA slope must be negative
        if slope >= 0:
            return 0.0

        # Fisher must be below threshold
        if fisher_val >= self.fisher_threshold:
            return 0.0

        # Fisher magnitude drives base signal
        fisher_strength = min(abs(fisher_val) / 3.0, 1.0)
        base = -0.3 - fisher_strength * 0.3  # -0.3 to -0.6

        # CMF confirmation
        if cmf_val < 0:
            cmf_boost = min(abs(cmf_val) * 0.5, 0.1)
            base -= cmf_boost

        return np.clip(base, -0.7, 0.0)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"HMA({self.hma_period})", "array": self._hma_line,
             "type": "overlay", "color": "#ffab40"},
            {"name": f"Fisher({self.fisher_period})", "array": self._fisher,
             "type": "subplot", "zero_line": True,
             "horizontal_lines": [self.fisher_threshold], "color": "#2196f3"},
            {"name": "Fisher Trigger", "array": self._fisher_trigger,
             "type": "subplot", "panel": f"Fisher({self.fisher_period})",
             "color": "#ff9800"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf,
             "type": "subplot", "zero_line": True, "color": "#ab47bc"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"HMA({self.hma_period})", datetimes, self._hma_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Fisher({self.fisher_period})",
                    [
                        self._make_subplot_trace("Fisher", datetimes, self._fisher, color="#2196f3"),
                        self._make_subplot_trace("Trigger", datetimes, self._fisher_trigger, color="#ff9800"),
                    ],
                    zero_line=True, horizontal_lines=[self.fisher_threshold],
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#ab47bc")],
                    zero_line=True,
                ),
            ],
        }
