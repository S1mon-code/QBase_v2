"""StrongTrendShortAG2hV7 — TEMA Slope + PPO Bearish + Chaikin Oscillator Negative.

Economic logic: TEMA(25) slope < 0 provides triple-smoothed trend direction
with minimal lag. PPO(10,25) < 0 confirms percentage-based momentum is negative.
Chaikin Oscillator < 0 validates that A/D line momentum is distribution-biased.
EMA(4) smoothing reduces overtrading.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.tema import tema
from indicators.momentum.ppo import ppo
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV7(TrendingStrategy):
    """TEMA(25) slope < 0 + PPO(10,25) < 0 + Chaikin Osc < 0.

    Signal logic (raw, pre-smoothing):
        All 3 conditions met -> -0.85
        TEMA slope < 0 AND PPO < 0 -> -0.55
        TEMA slope < 0 AND Chaikin < 0 -> -0.35
        else -> 0.0
    """

    name = "strong_trend_short_AG_2h_v7"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    tema_period: int = 25
    ppo_fast: int = 10
    ppo_slow: int = 25
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tema = tema(self._closes, period=self.tema_period)
        self._ppo_line, self._ppo_sig, self._ppo_hist = ppo(
            self._closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=9,
        )
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=3, slow=10,
        )

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        t = self._tema[i]
        t_prev = self._tema[i - 1] if i > 0 else np.nan
        p = self._ppo_line[i]
        ch = self._chaikin[i]

        if any(np.isnan(v) for v in (t, t_prev, p, ch)):
            return 0.0

        tema_slope_neg = t < t_prev
        ppo_neg = p < 0.0
        chaikin_neg = ch < 0.0

        if tema_slope_neg and ppo_neg and chaikin_neg:
            return -0.85
        if tema_slope_neg and ppo_neg:
            return -0.55
        if tema_slope_neg and chaikin_neg:
            return -0.35
        return 0.0

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema,
             "type": "overlay"},
            {"name": "PPO Line", "array": self._ppo_line, "panel": "PPO",
             "zero_line": True},
            {"name": "PPO Signal", "array": self._ppo_sig, "panel": "PPO"},
            {"name": "Chaikin Osc", "array": self._chaikin, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"TEMA({self.tema_period})", datetimes,
                               self._tema, color="#ffab40"),
        ]
        subplots = [
            self._make_subplot("PPO", [
                self._make_subplot_trace("PPO Line", datetimes, self._ppo_line,
                                         color="#42a5f5"),
                self._make_subplot_trace("PPO Signal", datetimes, self._ppo_sig,
                                         color="#ef5350"),
            ], zero_line=True),
            self._make_subplot("Chaikin Osc", [
                self._make_subplot_trace("Chaikin", datetimes, self._chaikin,
                                         color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
