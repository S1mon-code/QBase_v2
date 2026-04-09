"""MildTrendLongI2hV18 — KAMA Slope + CCI Direction.

Economic logic: Kaufman Adaptive Moving Average (KAMA) adapts its smoothing to
market noise — fast in trends, slow in chop. A rising KAMA confirms structural
trend direction. CCI measures price deviation from its statistical mean, with
positive CCI indicating bullish pressure. When KAMA slope and CCI agree, the
trend has both structural and statistical confirmation.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.kama import kama
from indicators.momentum.cci import cci
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV18(TrendingStrategy):
    """KAMA slope direction with CCI momentum confirmation.

    Signal logic:
        - KAMA rising AND CCI > 0 → +min(1.0, abs(CCI) / 200)
        - KAMA falling AND CCI < 0 → -min(1.0, abs(CCI) / 200)
        - Disagree → 0.0

    Attributes:
        kama_period:     KAMA efficiency ratio period.
        cci_period:      CCI calculation period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_2h_v18"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 40  # max(kama_period, cci_period) + 20

    kama_period: int = 20
    cci_period: int = 20
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
        """Precompute KAMA and CCI arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return KAMA slope signal confirmed by CCI direction."""
        if bar_index < 1:
            return 0.0

        kama_curr = self._kama[bar_index]
        kama_prev = self._kama[bar_index - 1]
        cci_val = self._cci[bar_index]

        if np.isnan(kama_curr) or np.isnan(kama_prev) or np.isnan(cci_val):
            return 0.0

        kama_rising = kama_curr > kama_prev
        kama_falling = kama_curr < kama_prev

        if kama_rising and cci_val > 0.0:
            return min(1.0, abs(cci_val) / 200.0)
        if kama_falling and cci_val < 0.0:
            return -min(1.0, abs(cci_val) / 200.0)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "kama", "params": {"period": self.kama_period}},
            {"name": "cci", "params": {"period": self.cci_period}},
        ]
