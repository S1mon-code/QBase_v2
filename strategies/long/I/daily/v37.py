"""MildTrendLongIDailyV37 — SuperTrend Fast + SuperTrend Slow Agreement (Multi-Timeframe).

Economic logic: A single SuperTrend can produce whipsaw signals in choppy
markets. Using two SuperTrend indicators with different ATR periods acts as
a multi-timeframe filter: the faster one is responsive to new trends, the
slower one confirms that the trend has sufficient structural backing. Only
when both agree is a full signal issued; disagreement means the trend is
contested and no position is taken.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV37(TrendingStrategy):
    """Dual SuperTrend agreement for confirmed trend direction.

    Signal logic:
        - Fast ST direction == 1  AND  Slow ST direction == 1:  +1.0
        - Fast ST direction == -1 AND  Slow ST direction == -1: -1.0
        - Disagreement (one up, one down):                        0.0

    Attributes:
        st_fast_period:  ATR period for the fast SuperTrend.
        st_slow_period:  ATR period for the slow SuperTrend.
        st_mult:         ATR multiplier shared by both SuperTrends.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v37"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 60  # st_slow_period(20) + st_slow_period(20) + 20

    st_fast_period: int = 10
    st_slow_period: int = 20
    st_mult: float = 3.0
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
        """Precompute fast and slow SuperTrend direction arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        _, self._st_fast_dir = supertrend(
            self._highs,
            self._lows,
            self._closes,
            period=self.st_fast_period,
            multiplier=self.st_mult,
        )
        _, self._st_slow_dir = supertrend(
            self._highs,
            self._lows,
            self._closes,
            period=self.st_slow_period,
            multiplier=self.st_mult,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return +1/-1 when both SuperTrends agree, else 0."""
        dir_fast = self._st_fast_dir[bar_index]
        dir_slow = self._st_slow_dir[bar_index]

        if np.isnan(dir_fast) or np.isnan(dir_slow):
            return 0.0

        if dir_fast == 1 and dir_slow == 1:
            return 1.0
        if dir_fast == -1 and dir_slow == -1:
            return -1.0
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "supertrend_fast",
                "params": {
                    "period": self.st_fast_period,
                    "multiplier": self.st_mult,
                },
            },
            {
                "name": "supertrend_slow",
                "params": {
                    "period": self.st_slow_period,
                    "multiplier": self.st_mult,
                },
            },
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"ST Dir({self.st_fast_period}/{self.st_slow_period})",
                [
                    self._make_subplot_trace("Fast Dir", datetimes, self._st_fast_dir, style="step", color="#bb86fc"),
                    self._make_subplot_trace("Slow Dir", datetimes, self._st_slow_dir, style="step", color="#4fc3f7"),
                ],
            )
        ]
        return {"overlays": [], "subplots": subplots}
