"""MildTrendLongI2hV24 — RMI + Volume Momentum.

Economic logic: Relative Momentum Index smooths RSI by comparing closes over
a lookback gap, reducing whipsaw. Volume Momentum (ratio > 1 means increasing
participation) acts as a conviction filter — strong RMI signals are amplified
when volume is expanding, dampened when it contracts.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.relative_momentum_index import rmi
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV24(TrendingStrategy):
    """RMI direction with Volume Momentum conviction filter.

    Signal logic:
        - RMI > 60 AND vol_mom > 1.1: +min(1.0, (RMI-50)/50)
        - RMI < 40 AND vol_mom > 1.1: -min(1.0, (50-RMI)/50)
        - vol_mom <= 1.1: signal * 0.3 (low conviction)
        - RMI in [40, 60]: 0.0
    """

    name = "mild_trend_long_I_2h_v24"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 48  # rmi_period(14) + vol_mom_period(14) + 20

    rmi_period: int = 14
    rmi_lookback: int = 5
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
        """Precompute RMI and Volume Momentum arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._rmi = rmi(
            self._closes, period=self.rmi_period, lookback=self.rmi_lookback,
        )
        self._vol_mom = volume_momentum(self._volumes, period=self.vol_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on RMI and Volume Momentum."""
        r = self._rmi[bar_index]
        vm = self._vol_mom[bar_index]

        if np.isnan(r) or np.isnan(vm):
            return 0.0

        # Neutral zone
        if 40 <= r <= 60:
            return 0.0

        if r > 60:
            raw = float(np.clip((r - 50) / 50, 0.0, 1.0))
            return raw if vm > 1.1 else raw * 0.3
        # r < 40
        raw = float(np.clip((50 - r) / 50, 0.0, 1.0))
        return -raw if vm > 1.1 else -raw * 0.3

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "rmi", "params": {"period": self.rmi_period, "lookback": self.rmi_lookback}},
            {"name": "volume_momentum", "params": {"period": self.vol_mom_period}},
        ]
