"""MildTrendLongIDailyV10 — Keltner(100,2.5) + MomentumAccel(80) + CMF(100).

Economic logic: Keltner Channel breakout identifies strong directional moves when
price exceeds the ATR-based band. Momentum acceleration (2nd derivative of price)
detects whether the trend is gaining steam or fading. CMF validates volume-weighted
buying pressure. Three signals prevent chasing exhausted trends.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.momentum_accel import momentum_acceleration
from indicators.trend.keltner import keltner
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV10(TrendingStrategy):
    name = "long_I_daily_v10"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup = 130

    kelt_period: int = 100
    kelt_mult: float = 2.5
    accel_fast: int = 40
    cmf_period: int = 100
    chandelier_mult: float = 2.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_period, atr_period=self.kelt_period // 2, multiplier=self.kelt_mult
        )
        self._accel = momentum_acceleration(self._closes, fast_period=self.accel_fast, slow_period=self.accel_fast)
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, period=self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        ku = self._kelt_upper[bar_index]
        km = self._kelt_mid[bar_index]
        acc = self._accel[bar_index]
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in [close, ku, km, acc, cmf_val]):
            return 0.0

        above_mid = close > km
        accel_positive = acc > 0.0
        cmf_positive = cmf_val > 0.0

        if not (above_mid and accel_positive and cmf_positive):
            return 0.0

        # Bonus for Keltner upper breakout
        breakout_bonus = 0.2 if close > ku else 0.0
        cmf_score = min(1.0, cmf_val / 0.3) * 0.3
        return min(1.0, 0.3 + breakout_bonus + cmf_score + 0.2)

    def get_indicator_config(self):
        return [
            {"name": "Keltner Upper", "array": self._kelt_upper, "type": "overlay", "style": "dash"},
            {"name": "Keltner Mid", "array": self._kelt_mid, "type": "overlay"},
            {"name": "Keltner Lower", "array": self._kelt_lower, "type": "overlay", "style": "dash"},
            {"name": "Mom Accel", "array": self._accel, "type": "subplot", "zero_line": True},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay("Keltner Upper", datetimes, self._kelt_upper, style="dash", color="#78909c"),
                self._make_overlay("Keltner Mid", datetimes, self._kelt_mid, color="#ffab40"),
                self._make_overlay("Keltner Lower", datetimes, self._kelt_lower, style="dash", color="#78909c"),
            ],
            "subplots": [
                self._make_subplot(
                    "Momentum Accel",
                    [self._make_subplot_trace("Accel", datetimes, self._accel, color="#bb86fc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "CMF",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#26a69a")],
                    zero_line=True,
                ),
            ],
        }
