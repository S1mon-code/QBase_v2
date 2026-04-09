"""MildTrendLongI4hV22 — HMA Slope + Klinger Oscillator.

Economic logic: Hull Moving Average responds quickly to trend changes with
minimal lag. Klinger Volume Oscillator measures volume-driven accumulation
vs distribution. Agreement indicates a volume-confirmed trend.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.volume.klinger import klinger
from indicators.volatility.atr import atr
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV22(TrendingStrategy):
    """HMA slope direction confirmed by Klinger Oscillator.

    Signal logic:
        - HMA rising AND KVO > signal -> +min(1.0, abs(slope)/atr * 5)
        - HMA falling AND KVO < signal -> -min(1.0, abs(slope)/atr * 5)
        - Disagreement -> 0.0
    """

    name = "long_I_4h_v22"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 94  # hma_period(40) + klinger_fast(34) + 20

    hma_period: int = 40
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
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma = hma(self._closes, self.hma_period)
        self._hma_slope = np.diff(self._hma, prepend=np.nan)
        self._kvo, self._kvo_signal = klinger(
            self._highs, self._lows, self._closes, self._volumes,
            fast=self.klinger_fast,
        )
        self._atr = atr(self._highs, self._lows, self._closes, period=14)

    def _generate_signal(self, bar_index: int) -> float:
        slope = self._hma_slope[bar_index]
        kvo_val = self._kvo[bar_index]
        kvo_sig = self._kvo_signal[bar_index]
        atr_val = self._atr[bar_index]

        if np.isnan(slope) or np.isnan(kvo_val) or np.isnan(kvo_sig) or np.isnan(atr_val):
            return 0.0
        if atr_val == 0:
            return 0.0

        strength = min(1.0, abs(slope) / atr_val * 5.0)

        if slope > 0 and kvo_val > kvo_sig:
            return strength
        if slope < 0 and kvo_val < kvo_sig:
            return -strength
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "hma", "params": {"period": self.hma_period}},
            {"name": "klinger", "params": {"fast": self.klinger_fast}},
            {"name": "atr", "params": {"period": 14}},
        ]
