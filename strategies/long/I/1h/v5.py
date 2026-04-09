"""MildTrendLongI1hV5 — KAMA(30) + MomentumAccel(20) + ChaikinOsc(10,30).

Economic logic: KAMA adapts to 1H iron ore efficiency ratio for smooth trend tracking.
Momentum acceleration (2nd derivative) detects whether trend is gaining steam.
Chaikin Oscillator measures A/D line momentum for accumulation validation. Signal
scales with acceleration positivity and Chaikin strength.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.momentum_accel import momentum_acceleration
from indicators.trend.kama import kama
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV5(TrendingStrategy):
    name = "long_I_1h_v5"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 50

    kama_period: int = 30
    accel_fast: int = 10
    accel_slow: int = 20
    chaikin_slow: int = 30
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._accel = momentum_acceleration(
            self._closes, fast_period=self.accel_fast, slow_period=self.accel_slow
        )
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=10, slow=self.chaikin_slow
        )

    def _generate_signal(self, bar_index: int) -> float:
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1] if bar_index > 0 else np.nan
        acc = self._accel[bar_index]
        ch = self._chaikin[bar_index]

        if any(np.isnan(v) for v in [k, k_prev, acc, ch]):
            return 0.0

        kama_rising = k > k_prev
        accel_positive = acc > 0.0
        chaikin_positive = ch > 0.0

        if not (kama_rising and accel_positive and chaikin_positive):
            return 0.0

        return min(1.0, 0.5 + 0.2)

    def get_indicator_config(self):
        return [
            {"name": f"KAMA({self.kama_period})", "array": self._kama, "type": "overlay"},
            {"name": "Mom Accel", "array": self._accel, "type": "subplot", "zero_line": True},
            {"name": "Chaikin Osc", "array": self._chaikin, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Momentum Accel",
                    [self._make_subplot_trace("Accel", datetimes, self._accel, color="#bb86fc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "Chaikin Oscillator",
                    [self._make_subplot_trace("Chaikin", datetimes, self._chaikin, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
