"""MildTrendLongIDailyV25 — Rate of Change × Volume Momentum.

Economic logic: Rate of Change measures the speed and magnitude of price moves
over a medium horizon. Volume Momentum (current volume relative to its EMA of
SMA) indicates whether participation is expanding behind the move. Strong ROC
with high volume momentum identifies high-conviction directional moves; weak
volume always reduces conviction to avoid low-quality signals.
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.volume.volume_momentum import volume_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV25(TrendingStrategy):
    """Rate of Change direction scaled by Volume Momentum.

    Signal logic:
        - abs(ROC) > 2.0 AND vol_mom > 1.1: roc_sign * 1.0 (strong signal)
        - abs(ROC) > 0   AND vol_mom > 1.0: roc_sign * 0.7 (moderate signal)
        - abs(ROC) > 0   AND vol_mom <= 1.0: roc_sign * 0.3 (weak, low volume)
        - ROC == 0: 0.0

    Attributes:
        roc_period:      Rate of Change lookback period.
        vol_mom_period:  Volume Momentum SMA and EMA period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v25"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 68  # roc_period(20) + vol_mom_period*2(28) + 20

    roc_period: int = 20
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
        """Precompute ROC and Volume Momentum arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._vol_mom = volume_momentum(self._volumes, period=self.vol_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal scaled by ROC magnitude and volume momentum level."""
        roc_val = self._roc[bar_index]
        vmom = self._vol_mom[bar_index]

        if np.isnan(roc_val) or np.isnan(vmom):
            return 0.0

        roc_sign = 1.0 if roc_val > 0 else -1.0 if roc_val < 0 else 0.0

        if abs(roc_val) > 2.0 and vmom > 1.1:
            return roc_sign * 1.0
        if abs(roc_val) > 0 and vmom > 1.0:
            return roc_sign * 0.7
        if abs(roc_val) > 0:
            return roc_sign * 0.3
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "rate_of_change", "params": {"period": self.roc_period}},
            {"name": "volume_momentum", "params": {"period": self.vol_mom_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"Vol Mom({self.vol_mom_period})",
                [self._make_subplot_trace("Vol Mom", datetimes, self._vol_mom, color="#bb86fc")],
                zero_line=True,
            ),
            self._make_subplot(
                f"ROC({self.roc_period})",
                [self._make_subplot_trace("ROC", datetimes, self._roc, color="#4fc3f7")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
