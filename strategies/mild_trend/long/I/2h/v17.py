"""MildTrendLongI2hV17 — SuperTrend + RSI Confirmation.

Economic logic: SuperTrend provides a clean binary trend direction using ATR-based
trailing stops. RSI confirms that momentum aligns with the trend — RSI above 50
in uptrends and below 50 in downtrends. The RSI distance from 50 scales signal
strength, giving stronger signals in high-conviction momentum regimes.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.momentum.rsi import rsi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV17(TrendingStrategy):
    """SuperTrend direction with RSI momentum confirmation.

    Signal logic:
        - SuperTrend bullish (dir=1) AND RSI > 50 → +min(1.0, (rsi-50)/50)
        - SuperTrend bearish (dir=-1) AND RSI < 50 → -min(1.0, (50-rsi)/50)
        - Disagree → 0.0

    Attributes:
        st_period:       SuperTrend ATR period.
        st_mult:         SuperTrend ATR multiplier.
        rsi_period:      RSI calculation period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_2h_v17"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 44  # st_period + rsi_period + 20

    st_period: int = 10
    st_mult: float = 3.0
    rsi_period: int = 14
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
        """Precompute SuperTrend and RSI arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_values, self._st_direction = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        self._rsi = rsi(self._closes, period=self.rsi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return SuperTrend directional signal confirmed by RSI."""
        st_dir = self._st_direction[bar_index]
        rsi_val = self._rsi[bar_index]

        if np.isnan(st_dir) or np.isnan(rsi_val):
            return 0.0

        if st_dir == 1.0 and rsi_val > 50.0:
            return min(1.0, (rsi_val - 50.0) / 50.0)
        if st_dir == -1.0 and rsi_val < 50.0:
            return -min(1.0, (50.0 - rsi_val) / 50.0)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "supertrend",
                "params": {"period": self.st_period, "multiplier": self.st_mult},
            },
            {"name": "rsi", "params": {"period": self.rsi_period}},
        ]
