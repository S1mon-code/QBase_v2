"""MildTrendLongI4hV25 — CCI + KAMA Slope.

Economic logic: Commodity Channel Index measures deviation from statistical
mean — extreme values indicate strong trending conditions. KAMA (Kaufman
Adaptive Moving Average) adapts its smoothing to market noise, providing
reliable slope signals in iron ore's volatile environment.
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.trend.kama import kama
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV25(TrendingStrategy):
    """CCI extremes confirmed by KAMA slope direction.

    Signal logic:
        - CCI > 100 AND KAMA rising -> +min(1.0, abs(CCI) / 200)
        - CCI < -100 AND KAMA falling -> -min(1.0, abs(CCI) / 200)
        - CCI in (0, 100) AND KAMA rising -> +0.4
        - CCI in (-100, 0) AND KAMA falling -> -0.4
        - Otherwise -> 0.0
    """

    name = "long_I_4h_v25"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 40  # max(cci_period, kama_period) + 20

    cci_period: int = 20
    kama_period: int = 20
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
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        self._kama = kama(self._closes, period=self.kama_period)
        self._kama_slope = np.diff(self._kama, prepend=np.nan)

    def _generate_signal(self, bar_index: int) -> float:
        cci_val = self._cci[bar_index]
        ks = self._kama_slope[bar_index]

        if np.isnan(cci_val) or np.isnan(ks):
            return 0.0

        kama_rising = ks > 0
        kama_falling = ks < 0

        if cci_val > 100 and kama_rising:
            return min(1.0, abs(cci_val) / 200.0)
        if cci_val < -100 and kama_falling:
            return -min(1.0, abs(cci_val) / 200.0)
        if 0 < cci_val <= 100 and kama_rising:
            return 0.4
        if -100 <= cci_val < 0 and kama_falling:
            return -0.4
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "cci", "params": {"period": self.cci_period}},
            {"name": "kama", "params": {"period": self.kama_period}},
        ]
