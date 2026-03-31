"""Trend Medium V1 — SuperTrend + Volume Momentum.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: Trend confirmed by institutional volume flow.  SuperTrend
identifies the dominant direction while Volume Momentum filters out
low-conviction moves where price trends without volume support — a common
sign of retail-driven noise in Chinese futures markets.

Who loses money: Retail traders who chase breakouts without volume
confirmation and get caught in false moves.
"""

from __future__ import annotations

import numpy as np

from strategies.templates.trending_template import TrendingStrategy
from indicators.trend.supertrend import supertrend
from indicators.volume.volume_momentum import volume_momentum


class TrendMediumV1(TrendingStrategy):
    """SuperTrend direction + Volume Momentum confirmation.

    Signal is +/-1 when SuperTrend direction aligns with elevated volume
    momentum; signal is halved when volume momentum is below threshold.

    Attributes:
        st_period:          SuperTrend ATR lookback.
        st_mult:            SuperTrend ATR multiplier.
        vol_mom_period:     Volume Momentum lookback.
        vol_mom_threshold:  Minimum VM ratio for full conviction.
        chandelier_mult:    Chandelier Exit multiplier (optimisable).
    """

    name = "trend_medium_v1"
    horizon = "medium"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 150

    st_period: int = 10
    st_mult: float = 3.0
    vol_mom_period: int = 20
    vol_mom_threshold: float = 1.5
    chandelier_mult: float = 2.5

    # --- Precomputed arrays ---
    _st_direction: np.ndarray | None = None
    _vol_mom: np.ndarray | None = None

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
        """Precompute SuperTrend direction and Volume Momentum arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        _st_line, self._st_direction = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        self._vol_mom = volume_momentum(
            self._volumes, period=self.vol_mom_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Combine SuperTrend direction with Volume Momentum."""
        direction = self._st_direction[bar_index]
        vm = self._vol_mom[bar_index]

        if np.isnan(direction) or np.isnan(vm):
            return 0.0

        # Full signal when volume momentum exceeds threshold
        if vm >= self.vol_mom_threshold:
            return float(direction)

        # Half signal on weak volume
        return float(direction) * 0.5

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "supertrend",
                "params": {"period": self.st_period, "multiplier": self.st_mult},
            },
            {
                "name": "volume_momentum",
                "params": {"period": self.vol_mom_period},
            },
        ]
