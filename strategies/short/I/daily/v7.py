"""MildTrendShortIDailyV7 — KAMA Slope + PPO Negative + Chaikin Oscillator.

Economic logic: KAMA(35) slope turning negative shows the adaptive average
is declining, filtering noise better than raw EMA. PPO below zero confirms
percentage-normalized bearish momentum. Negative Chaikin Oscillator validates
that accumulation/distribution momentum is bearish.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.momentum.ppo import ppo
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from indicators.trend.linear_regression import linear_regression_slope
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV7(TrendingStrategy):
    """KAMA(35) slope < 0 + PPO(10,30) < 0 + Chaikin Osc < 0."""

    name = "short_I_daily_v7"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 55

    # Optimizable parameters (<=5 including chandelier_mult)
    kama_period: int = 35
    ppo_fast: int = 10
    ppo_slow: int = 30
    chaikin_slow: int = 10
    chandelier_mult: float = 4.0

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
        self._kama_slope = linear_regression_slope(self._kama_line, 10)
        self._ppo_line, self._ppo_signal, self._ppo_hist = ppo(
            self._closes, self.ppo_fast, self.ppo_slow, 9,
        )
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=3, slow=self.chaikin_slow,
        )

        # Pre-smooth raw signal
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        slope = self._kama_slope[bar_index]
        ppo_val = self._ppo_line[bar_index]
        chaikin_val = self._chaikin[bar_index]

        if any(np.isnan(v) for v in (slope, ppo_val, chaikin_val)):
            return 0.0

        # KAMA slope must be negative
        if slope >= 0:
            return 0.0

        # PPO must be negative
        if ppo_val >= 0:
            return 0.0

        # PPO magnitude drives base signal
        ppo_strength = min(abs(ppo_val) / 3.0, 1.0)
        base = -0.3 - ppo_strength * 0.3  # -0.3 to -0.6

        # Chaikin confirmation
        if chaikin_val < 0:
            base -= 0.1

        return np.clip(base, -0.7, 0.0)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama_line,
             "type": "overlay", "color": "#ffab40"},
            {"name": f"PPO({self.ppo_fast},{self.ppo_slow})", "array": self._ppo_line,
             "type": "subplot", "zero_line": True, "color": "#2196f3"},
            {"name": "PPO Signal", "array": self._ppo_signal,
             "type": "subplot", "panel": f"PPO({self.ppo_fast},{self.ppo_slow})",
             "color": "#ff9800"},
            {"name": "Chaikin Osc", "array": self._chaikin,
             "type": "subplot", "zero_line": True, "color": "#ab47bc"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"PPO({self.ppo_fast},{self.ppo_slow})",
                    [
                        self._make_subplot_trace("PPO", datetimes, self._ppo_line, color="#2196f3"),
                        self._make_subplot_trace("Signal", datetimes, self._ppo_signal, color="#ff9800"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "Chaikin Osc",
                    [self._make_subplot_trace("Chaikin", datetimes, self._chaikin, color="#ab47bc")],
                    zero_line=True,
                ),
            ],
        }
