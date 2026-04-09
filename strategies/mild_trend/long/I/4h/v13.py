"""MildTrendLongI4hV13 — SuperTrend + OI Flow.

Economic logic: SuperTrend provides clean trend direction with ATR-based trailing
stops.  OI Flow measures whether new money is flowing into the trend direction.
When SuperTrend flips direction and open interest confirms smart money is adding
positions, the move has institutional conviction behind it.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV13(TrendingStrategy):
    """SuperTrend direction confirmed by OI Flow momentum.

    Signal logic:
        ST direction = 1 AND OI flow > signal → +1.0
        ST direction = -1 AND flow < signal  → -1.0
        ST agree but no OI confirm           → 0.4 * direction
        Else → 0.0
    """

    name = "mild_trend_long_I_4h_v13"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 48  # st_period + oi_period + 20

    # Optimizable parameters
    st_period: int = 14
    st_mult: float = 3.0
    oi_period: int = 14
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
        """Precompute SuperTrend and OI Flow arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_values, self._st_dir = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        self._oi_flow, self._oi_signal = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from SuperTrend direction filtered by OI Flow."""
        st_d = self._st_dir[bar_index]
        flow = self._oi_flow[bar_index]
        sig = self._oi_signal[bar_index]

        if np.isnan(st_d) or np.isnan(flow) or np.isnan(sig):
            return 0.0

        direction = float(st_d)  # 1.0 or -1.0

        if direction == 1.0 and flow > sig:
            return 1.0
        if direction == -1.0 and flow < sig:
            return -1.0
        if direction == 1.0:
            return 0.4
        if direction == -1.0:
            return -0.4

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "supertrend", "params": {"period": self.st_period, "multiplier": self.st_mult}},
            {"name": "oi_flow", "params": {"period": self.oi_period}},
        ]
