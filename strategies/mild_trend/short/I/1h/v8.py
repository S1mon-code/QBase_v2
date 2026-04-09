"""MildTrendShortI1hV8 — Linear Regression slope + PPO negative + EMV negative.

Economic logic: Linear Regression(20) negative slope on 1H provides statistically
fitted downtrend direction. PPO(6,18) < 0 confirms percentage-based momentum is
bearish. EMV(12) < 0 validates that price movement relative to volume favours
sellers. Signal smoothed with EMA(3) for mild-regime stability.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression_slope
from indicators.momentum.ppo import ppo
from indicators.volume.emv import emv
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV8(TrendingStrategy):
    """Linear Reg(20) slope < 0 + PPO(6,18) < 0 + EMV(12) < 0."""

    name = "mild_trend_short_I_1h_v8"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 35

    lr_period: int = 20
    ppo_fast: int = 6
    ppo_slow: int = 18
    emv_period: int = 12
    chandelier_mult: float = 3.1

    _lr_slope: np.ndarray | None = None
    _ppo_line: np.ndarray | None = None
    _ppo_sig: np.ndarray | None = None
    _ppo_hist: np.ndarray | None = None
    _emv_arr: np.ndarray | None = None
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
        self._lr_slope = linear_regression_slope(self._closes, self.lr_period)
        self._ppo_line, self._ppo_sig, self._ppo_hist = ppo(
            self._closes, self.ppo_fast, self.ppo_slow, 9,
        )
        self._emv_arr = emv(self._highs, self._lows, self._volumes, self.emv_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        sl = self._lr_slope[bar_index]
        pp = self._ppo_line[bar_index]
        ev = self._emv_arr[bar_index]

        if any(np.isnan(v) for v in (sl, pp, ev)):
            return 0.0

        if sl >= 0:
            return 0.0

        signal = -0.3

        if pp < 0:
            signal -= min(0.2, abs(pp) * 0.1)

        if ev < 0:
            signal -= 0.15

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"LR Slope({self.lr_period})", "array": self._lr_slope, "type": "subplot", "zero_line": True},
            {"name": "PPO", "array": self._ppo_line, "type": "subplot", "panel": "PPO", "zero_line": True},
            {"name": "PPO Hist", "array": self._ppo_hist, "type": "subplot", "panel": "PPO",
             "style": "bar", "color_positive": "#26a69a", "color_negative": "#ef5350"},
            {"name": f"EMV({self.emv_period})", "array": self._emv_arr, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"LR Slope({self.lr_period})",
                    [self._make_subplot_trace("Slope", datetimes, self._lr_slope, color="#ab47bc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "PPO",
                    [
                        self._make_subplot_trace("PPO", datetimes, self._ppo_line, color="#bb86fc"),
                        self._make_subplot_trace("Hist", datetimes, self._ppo_hist, style="bar",
                                                 color_positive="#26a69a", color_negative="#ef5350"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"EMV({self.emv_period})",
                    [self._make_subplot_trace("EMV", datetimes, self._emv_arr, color="#ff7043")],
                    zero_line=True,
                ),
            ],
        }
