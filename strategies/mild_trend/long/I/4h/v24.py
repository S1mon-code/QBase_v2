"""MildTrendLongI4hV24 — DEMA Cross + OI Flow Confirmation.

Economic logic: Double EMA crossover captures trend shifts with less lag than
SMA. Open Interest flow measures whether new money is entering in the trend
direction, confirming institutional participation.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.dema import dema
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV24(TrendingStrategy):
    """DEMA fast/slow crossover confirmed by OI Flow.

    Signal logic:
        - DEMA_fast > DEMA_slow AND OI flow > signal -> +1.0
        - DEMA_fast < DEMA_slow AND flow < signal -> -1.0
        - DEMA agrees but no OI confirm -> +/-0.4
        - Otherwise -> 0.0
    """

    name = "mild_trend_long_I_4h_v24"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 134  # dema_slow(100) + oi_period(14) + 20

    dema_fast: int = 50
    dema_slow: int = 100
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
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dema_f = dema(self._closes, self.dema_fast)
        self._dema_s = dema(self._closes, self.dema_slow)
        self._oi_flow, self._oi_signal = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        df = self._dema_f[bar_index]
        ds = self._dema_s[bar_index]
        of_val = self._oi_flow[bar_index]
        os_val = self._oi_signal[bar_index]

        if np.isnan(df) or np.isnan(ds) or np.isnan(of_val) or np.isnan(os_val):
            return 0.0

        dema_bull = df > ds
        dema_bear = df < ds
        oi_bull = of_val > os_val
        oi_bear = of_val < os_val

        if dema_bull and oi_bull:
            return 1.0
        if dema_bear and oi_bear:
            return -1.0
        if dema_bull:
            return 0.4
        if dema_bear:
            return -0.4
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "dema", "params": {"fast": self.dema_fast, "slow": self.dema_slow}},
            {"name": "oi_flow", "params": {"period": self.oi_period}},
        ]
