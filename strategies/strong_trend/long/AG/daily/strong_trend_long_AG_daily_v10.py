"""StrongTrendLongAGDailyV10 — Keltner(120,3.0) + MomentumAccel(100) + ChaikinOsc(30,80).

Economic logic: Wide Keltner Channel (3.0x ATR) captures Silver's volatile
daily trading range. Momentum Acceleration (2nd derivative) detects when
AG's trend is accelerating vs decelerating. Chaikin Oscillator measures the
momentum of accumulation/distribution flow.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.keltner import keltner
from indicators.momentum.momentum_accel import momentum_acceleration
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAGDailyV10(TrendingStrategy):
    """Keltner channel + Momentum Acceleration + Chaikin Oscillator.

    Signal logic:
        - Price > Keltner mid AND MomAccel > 0 AND ChaikinOsc > 0 → long
        - Strength scales with proximity to upper band and Chaikin magnitude

    Attributes:
        kelt_ema:        Keltner EMA period.
        kelt_mult:       Keltner ATR multiplier (wide for AG).
        mom_fast:        Momentum Acceleration fast period.
        mom_slow:        Momentum Acceleration slow period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "strong_trend_long_AG_daily_v10"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 200

    kelt_ema: int = 120
    kelt_mult: float = 3.0
    mom_fast: int = 50
    mom_slow: int = 100
    chandelier_mult: float = 3.5

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
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_ema, multiplier=self.kelt_mult,
        )
        self._mom_accel = momentum_acceleration(
            self._closes, fast_period=self.mom_fast, slow_period=self.mom_slow,
        )
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            fast=30, slow=80,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        km = self._kelt_mid[bar_index]
        ku = self._kelt_upper[bar_index]
        ma = self._mom_accel[bar_index]
        ch = self._chaikin[bar_index]

        if any(np.isnan(v) for v in (close, km, ku, ma, ch)):
            return 0.0

        above_mid = close > km
        accel_pos = ma > 0.0
        chaikin_pos = ch > 0.0

        if above_mid and accel_pos and chaikin_pos:
            band_width = ku - km if ku > km else 1.0
            pos_in_band = min(1.0, (close - km) / band_width)
            strength = min(1.0, pos_in_band * 0.5 + 0.4)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"Keltner Upper({self.kelt_ema},{self.kelt_mult})",
             "array": self._kelt_upper, "type": "overlay"},
            {"name": f"Keltner Mid({self.kelt_ema})",
             "array": self._kelt_mid, "type": "overlay"},
            {"name": f"Keltner Lower({self.kelt_ema},{self.kelt_mult})",
             "array": self._kelt_lower, "type": "overlay"},
            {"name": f"MomAccel({self.mom_fast},{self.mom_slow})",
             "array": self._mom_accel, "type": "subplot", "zero_line": True},
            {"name": "ChaikinOsc(30,80)", "array": self._chaikin, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"Keltner Upper", datetimes, self._kelt_upper, color="#66bb6a"),
                self._make_overlay(f"Keltner Mid", datetimes, self._kelt_mid, style="dash", color="#ffab40"),
                self._make_overlay(f"Keltner Lower", datetimes, self._kelt_lower, color="#ef5350"),
            ],
            "subplots": [
                self._make_subplot(
                    f"MomAccel({self.mom_fast},{self.mom_slow})",
                    [self._make_subplot_trace("MomAccel", datetimes, self._mom_accel, color="#ab47bc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "ChaikinOsc(30,80)",
                    [self._make_subplot_trace("Chaikin", datetimes, self._chaikin, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
