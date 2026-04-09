"""MildTrendLongI4hV26 — SuperTrend Direction + Volume Momentum.

Economic logic: SuperTrend provides a clean trend direction with built-in
volatility adaptation via ATR. Volume Momentum above 1.0 means current volume
exceeds its recent average — confirming that the trend has participation.
High-volume trends in iron ore are more sustainable.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV26(TrendingStrategy):
    """SuperTrend direction confirmed by Volume Momentum.

    Signal logic:
        - ST direction = 1 AND vol_mom > 1.1 -> +1.0
        - ST direction = -1 AND vol_mom > 1.1 -> -1.0
        - ST has direction but vol_mom <= 1.1 -> 0.3 * direction
        - Otherwise -> 0.0
    """

    name = "long_I_4h_v26"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 48  # st_period(14) + vol_mom_period(14) + 20

    st_period: int = 14
    st_mult: float = 3.0
    vol_mom_period: int = 14
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
        self._st_values, self._st_dir = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        self._vol_mom = volume_momentum(self._volumes, period=self.vol_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        direction = self._st_dir[bar_index]
        vm = self._vol_mom[bar_index]

        if np.isnan(direction) or np.isnan(vm):
            return 0.0

        d = int(direction)
        if d == 0:
            return 0.0

        if vm > 1.1:
            return float(d)  # +1.0 or -1.0
        return 0.3 * d

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "supertrend", "params": {"period": self.st_period, "multiplier": self.st_mult}},
            {"name": "volume_momentum", "params": {"period": self.vol_mom_period}},
        ]
