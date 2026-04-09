"""StrongTrendLongAG4hV2 — SuperTrend(40,3.2) + MACD(18,45,14) + VolumeSpike(30).

Economic logic: SuperTrend with 3.2x multiplier captures AG's 4H trend while
avoiding false breakouts in volatile sessions. MACD measures intermediate
momentum. Volume spike detection confirms breakout conviction.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.momentum.macd import macd
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAG4hV2(TrendingStrategy):
    name = "strong_trend_long_AG_4h_v2"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    st_period: int = 40
    st_mult: float = 3.2
    macd_fast: int = 18
    macd_slow: int = 45
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        self._macd_line, self._macd_signal, _ = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow, signal=14,
        )
        self._vol_spike = volume_spike(self._volumes, period=30, threshold=2.0)

    def _generate_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        ml = self._macd_line[bar_index]
        ms = self._macd_signal[bar_index]

        if np.isnan(st_d) or np.isnan(ml) or np.isnan(ms):
            return 0.0

        st_bull = st_d > 0
        macd_bull = ml > ms
        spike = bool(self._vol_spike[bar_index]) if bar_index < len(self._vol_spike) else False

        if st_bull and macd_bull:
            base = min(1.0, abs(ml - ms) / (abs(self._closes[bar_index]) * 0.01 + 1e-9) + 0.3)
            if spike:
                base = min(1.0, base + 0.2)
            return max(0.0, base)
        return 0.0

    def get_indicator_config(self):
        return [
            {"name": f"SuperTrend({self.st_period},{self.st_mult})",
             "array": self._st_line, "type": "overlay"},
            {"name": f"MACD({self.macd_fast},{self.macd_slow})",
             "array": self._macd_line, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period},{self.st_mult})",
                                   datetimes, self._st_line, color="#ff7043"),
            ],
            "subplots": [
                self._make_subplot(
                    f"MACD({self.macd_fast},{self.macd_slow})",
                    [self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._macd_signal, color="#ff8a65")],
                    zero_line=True,
                ),
            ],
        }
