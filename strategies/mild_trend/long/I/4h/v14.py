"""MildTrendLongI4hV14 — ALMA + Klinger Volume Oscillator.

Economic logic: ALMA (Arnaud Legoux Moving Average) provides smooth trend detection
with minimal lag via its Gaussian-weighted kernel.  Klinger Volume Oscillator
measures the volume force behind price movements.  When price is above ALMA and
Klinger confirms accumulation, the uptrend has volume support.  Without Klinger
confirmation, the price move may be a low-conviction drift.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.alma import alma
from indicators.volume.klinger import klinger
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV14(TrendingStrategy):
    """ALMA trend direction confirmed by Klinger Volume Oscillator.

    Signal logic:
        Close > ALMA AND KVO > KVO signal → +1.0
        Close < ALMA AND KVO < signal     → -1.0
        Direction but no volume confirm   → 0.3 * sign
        Else → 0.0
    """

    name = "mild_trend_long_I_4h_v14"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 74  # alma_period + klinger_fast + 20

    # Optimizable parameters
    alma_period: int = 20
    alma_offset: float = 0.85
    klinger_fast: int = 34
    chandelier_mult: float = 2.5

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
        """Precompute ALMA and Klinger arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._alma = alma(self._closes, period=self.alma_period, offset=self.alma_offset)
        self._kvo, self._kvo_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from ALMA trend filtered by Klinger volume."""
        c = self._closes[bar_index]
        alma_val = self._alma[bar_index]
        kvo_val = self._kvo[bar_index]
        kvo_sig = self._kvo_signal[bar_index]

        if np.isnan(c) or np.isnan(alma_val) or np.isnan(kvo_val) or np.isnan(kvo_sig):
            return 0.0

        price_bull = c > alma_val
        price_bear = c < alma_val
        vol_bull = kvo_val > kvo_sig
        vol_bear = kvo_val < kvo_sig

        if price_bull and vol_bull:
            return 1.0
        if price_bear and vol_bear:
            return -1.0
        if price_bull:
            return 0.3
        if price_bear:
            return -0.3

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "alma", "params": {"period": self.alma_period, "offset": self.alma_offset}},
            {"name": "klinger", "params": {"fast": self.klinger_fast}},
        ]
