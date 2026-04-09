"""StrongTrendLongAG1hV9 — Keltner(28,2.2) + TSI(18,10) + Klinger(40).

Economic logic: Keltner Channel with moderate multiplier captures AG's 1H
volatility envelope. TSI with fast parameters detects momentum shifts quickly.
Klinger Volume Oscillator (substituted for SmartMoney) tracks institutional
accumulation on 1H Silver bars.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.keltner import keltner
from indicators.momentum.tsi import tsi
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG1hV9(TrendingStrategy):
    name = "strong_trend_long_AG_1h_v9"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    kelt_ema: int = 28
    kelt_mult: float = 2.2
    tsi_long: int = 18
    tsi_short: int = 10
    chandelier_mult: float = 2.8

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_ema, multiplier=self.kelt_mult,
        )
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=self.tsi_short,
        )
        self._klinger_line, self._klinger_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=20, slow=40, signal=13,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        km = self._kelt_mid[bar_index]
        t = self._tsi_line[bar_index]
        kl = self._klinger_line[bar_index]
        ks = self._klinger_signal[bar_index]

        if any(np.isnan(v) for v in (close, km, t, kl, ks)):
            return 0.0

        if close > km and t > 0.0 and kl > ks:
            strength = min(1.0, t / 20.0 * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"Keltner Upper({self.kelt_ema})", "array": self._kelt_upper, "type": "overlay"},
            {"name": f"Keltner Mid({self.kelt_ema})", "array": self._kelt_mid, "type": "overlay"},
            {"name": f"Keltner Lower({self.kelt_ema})", "array": self._kelt_lower, "type": "overlay"},
            {"name": f"TSI({self.tsi_long},{self.tsi_short})", "array": self._tsi_line,
             "type": "subplot", "zero_line": True},
            {"name": "Klinger", "array": self._klinger_line,
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
                    f"TSI({self.tsi_long},{self.tsi_short})",
                    [self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._tsi_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "Klinger",
                    [self._make_subplot_trace("Klinger", datetimes, self._klinger_line, color="#42a5f5"),
                     self._make_subplot_trace("Signal", datetimes, self._klinger_signal, color="#ff8a65")],
                    zero_line=True,
                ),
            ],
        }
