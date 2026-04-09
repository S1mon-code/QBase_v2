"""MildTrendLongI4hV6 — EMA(35) + CCI(40) + ChaikinOsc(20,50).

Economic logic: EMA provides smooth trend direction for 4H iron ore. CCI above zero
confirms price is above its statistical mean — bullish momentum. Chaikin Oscillator
measures A/D line momentum, validating accumulation. Signal scales with CCI magnitude
and Chaikin strength for gradual entry.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.trend.ema import ema
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV6(TrendingStrategy):
    name = "long_I_4h_v6"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 60

    ema_period: int = 35
    cci_period: int = 40
    chaikin_fast: int = 20
    chaikin_slow: int = 50
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema = ema(self._closes, period=self.ema_period)
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.chaikin_fast, slow=self.chaikin_slow
        )

    def _generate_signal(self, bar_index: int) -> float:
        e = self._ema[bar_index]
        e_prev = self._ema[bar_index - 1] if bar_index > 0 else np.nan
        cci_val = self._cci[bar_index]
        ch = self._chaikin[bar_index]
        close = self._closes[bar_index]

        if any(np.isnan(v) for v in [e, e_prev, cci_val, ch, close]):
            return 0.0

        ema_rising = e > e_prev
        cci_bullish = cci_val > 0.0
        chaikin_positive = ch > 0.0

        if not (ema_rising and cci_bullish and chaikin_positive):
            return 0.0

        cci_score = min(1.0, cci_val / 150.0) * 0.5
        return min(1.0, 0.3 + cci_score + 0.2)

    def get_indicator_config(self):
        return [
            {"name": f"EMA({self.ema_period})", "array": self._ema, "type": "overlay"},
            {"name": f"CCI({self.cci_period})", "array": self._cci, "type": "subplot", "zero_line": True},
            {"name": "Chaikin Osc", "array": self._chaikin, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_period})", datetimes, self._ema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "CCI",
                    [self._make_subplot_trace("CCI", datetimes, self._cci, color="#42a5f5")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "Chaikin Oscillator",
                    [self._make_subplot_trace("Chaikin", datetimes, self._chaikin, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
