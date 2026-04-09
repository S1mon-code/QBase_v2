"""StrongTrendLongAG4hV9 — Keltner(40,2.5) + ROC(25) + ChaikinOsc(15,40).

Economic logic: Keltner Channel with 2.5x ATR captures AG's 4H volatility
envelope. Rate of Change provides raw momentum magnitude. Chaikin Oscillator
measures acceleration of accumulation/distribution.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.keltner import keltner
from indicators.momentum.roc import rate_of_change
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG4hV9(TrendingStrategy):
    name = "strong_trend_long_AG_4h_v9"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    kelt_ema: int = 40
    kelt_mult: float = 2.5
    roc_period: int = 25
    chaikin_fast: int = 15
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_ema, multiplier=self.kelt_mult,
        )
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.chaikin_fast, slow=40,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        km = self._kelt_mid[bar_index]
        roc = self._roc[bar_index]
        ch = self._chaikin[bar_index]

        if any(np.isnan(v) for v in (close, km, roc, ch)):
            return 0.0

        if close > km and roc > 0.0 and ch > 0.0:
            strength = min(1.0, roc / 5.0 * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"Keltner Upper({self.kelt_ema})", "array": self._kelt_upper, "type": "overlay"},
            {"name": f"Keltner Mid({self.kelt_ema})", "array": self._kelt_mid, "type": "overlay"},
            {"name": f"Keltner Lower({self.kelt_ema})", "array": self._kelt_lower, "type": "overlay"},
            {"name": f"ROC({self.roc_period})", "array": self._roc,
             "type": "subplot", "zero_line": True},
            {"name": f"ChaikinOsc({self.chaikin_fast},40)", "array": self._chaikin,
             "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("Keltner Upper", datetimes, self._kelt_upper, color="#66bb6a"),
                self._make_overlay("Keltner Mid", datetimes, self._kelt_mid, style="dash", color="#ffab40"),
                self._make_overlay("Keltner Lower", datetimes, self._kelt_lower, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    f"ROC({self.roc_period})",
                    [self._make_subplot_trace("ROC", datetimes, self._roc, color="#ab47bc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"ChaikinOsc({self.chaikin_fast},40)",
                    [self._make_subplot_trace("Chaikin", datetimes, self._chaikin, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
