"""MildTrendLongI4hV29 — Mass Index Reversal Bulge + MACD Direction.

Economic logic: Mass Index detects range expansion that often precedes trend
reversals. A "reversal bulge" (mass > 27 then drops below 26.5) signals an
impending direction change. MACD determines which direction the reversal
resolves to — critical for catching iron ore trend turns early.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.mass_index import mass_index
from indicators.momentum.macd import macd
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV29(TrendingStrategy):
    """Mass Index reversal bulge with MACD directional filter.

    Signal logic:
        - Reversal bulge AND MACD > 0 -> +1.0 (reversal up)
        - Reversal bulge AND MACD < 0 -> -1.0 (reversal down)
        - No bulge: MACD > 0 -> +0.3, MACD < 0 -> -0.3
    """

    name = "mild_trend_long_I_4h_v29"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 71  # mass_sum(25) + macd_slow(26) + 20

    mass_ema: int = 9
    mass_sum: int = 25
    macd_fast: int = 12
    macd_slow: int = 26
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
        self._mass = mass_index(
            self._highs, self._lows,
            ema_period=self.mass_ema, sum_period=self.mass_sum,
        )
        self._macd_line, _, _ = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow,
        )
        # Precompute reversal bulge: mass[i-1] > 27 AND mass[i] < 26.5
        n = len(self._mass)
        self._bulge = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not np.isnan(self._mass[i - 1]) and not np.isnan(self._mass[i]):
                self._bulge[i] = self._mass[i - 1] > 27.0 and self._mass[i] < 26.5

    def _generate_signal(self, bar_index: int) -> float:
        macd_val = self._macd_line[bar_index]

        if np.isnan(macd_val):
            return 0.0

        if self._bulge[bar_index]:
            if macd_val > 0:
                return 1.0
            if macd_val < 0:
                return -1.0

        if macd_val > 0:
            return 0.3
        if macd_val < 0:
            return -0.3
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "mass_index", "params": {"ema_period": self.mass_ema, "sum_period": self.mass_sum}},
            {"name": "macd", "params": {"fast": self.macd_fast, "slow": self.macd_slow}},
        ]
