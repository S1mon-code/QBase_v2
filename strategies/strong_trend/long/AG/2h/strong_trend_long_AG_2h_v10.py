"""StrongTrendLongAG2hV10 — Keltner(55,2.5) + MomentumAccel(35) + ChaikinOsc(15,45).

Economic logic: Keltner Channel with 2.5x ATR captures AG's 2H volatility
envelope. Momentum Acceleration detects trend acceleration phase. Chaikin
Oscillator measures A/D line momentum for volume confirmation.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.keltner import keltner
from indicators.momentum.momentum_accel import momentum_acceleration
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG2hV10(TrendingStrategy):
    name = "strong_trend_long_AG_2h_v10"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    kelt_ema: int = 55
    kelt_mult: float = 2.5
    mom_fast: int = 15
    mom_slow: int = 35
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_ema, multiplier=self.kelt_mult,
        )
        self._mom_accel = momentum_acceleration(
            self._closes, fast_period=self.mom_fast, slow_period=self.mom_slow,
        )
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=15, slow=45,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        km = self._kelt_mid[bar_index]
        ku = self._kelt_upper[bar_index]
        ma = self._mom_accel[bar_index]
        ch = self._chaikin[bar_index]

        if any(np.isnan(v) for v in (close, km, ku, ma, ch)):
            return 0.0

        if close > km and ma > 0.0 and ch > 0.0:
            band_w = ku - km if ku > km else 1.0
            pos = min(1.0, (close - km) / band_w)
            strength = min(1.0, pos * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"Keltner Upper({self.kelt_ema})", "array": self._kelt_upper, "type": "overlay"},
            {"name": f"Keltner Mid({self.kelt_ema})", "array": self._kelt_mid, "type": "overlay"},
            {"name": f"Keltner Lower({self.kelt_ema})", "array": self._kelt_lower, "type": "overlay"},
            {"name": f"MomAccel({self.mom_fast},{self.mom_slow})", "array": self._mom_accel,
             "type": "subplot", "zero_line": True},
            {"name": "ChaikinOsc(15,45)", "array": self._chaikin,
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
                    f"MomAccel({self.mom_fast},{self.mom_slow})",
                    [self._make_subplot_trace("MomAccel", datetimes, self._mom_accel, color="#ab47bc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "ChaikinOsc(15,45)",
                    [self._make_subplot_trace("Chaikin", datetimes, self._chaikin, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
