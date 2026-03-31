"""Trend Medium V4 — KAMA Crossover + Volume Efficiency.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: KAMA (Kaufman Adaptive Moving Average) adapts its speed to
market efficiency.  A fast KAMA crossing above a slow KAMA signals a trend
with genuine price movement efficiency.  Volume Efficiency filters confirm
that the move has institutional weight (high price movement per unit of
volume, indicating conviction).

Who loses money: Low-conviction breakout traders who trade noisy price moves
with low volume-to-price conversion.
"""

from __future__ import annotations

import numpy as np

from strategies.templates.trending_template import TrendingStrategy
from indicators.trend.kama import kama
from indicators.volume.volume_efficiency import volume_efficiency


class TrendMediumV4(TrendingStrategy):
    """Fast KAMA / Slow KAMA crossover + Volume Efficiency z-score filter.

    Full signal when fast KAMA crosses slow KAMA and volume efficiency z-score
    is above threshold; half signal when KAMA direction is clear but efficiency
    is below threshold.

    Attributes:
        kama_fast_period:   KAMA period for the fast line.
        kama_slow_period:   KAMA period for the slow line.
        vol_eff_period:     Volume Efficiency lookback period.
        vol_eff_threshold:  Minimum efficiency z-score for full conviction.
        chandelier_mult:    Chandelier Exit multiplier (optimisable).
    """

    name = "trend_medium_v4"
    horizon = "medium"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 120

    kama_fast_period: int = 10
    kama_slow_period: int = 30
    vol_eff_period: int = 20
    vol_eff_threshold: float = 0.0
    chandelier_mult: float = 2.5

    # --- Precomputed arrays ---
    _fast_kama: np.ndarray | None = None
    _slow_kama: np.ndarray | None = None
    _vol_eff_z: np.ndarray | None = None

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
        """Precompute fast KAMA, slow KAMA, and Volume Efficiency z-score arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._fast_kama = kama(self._closes, period=self.kama_fast_period)
        self._slow_kama = kama(self._closes, period=self.kama_slow_period)
        _, self._vol_eff_z = volume_efficiency(
            self._closes, self._volumes, period=self.vol_eff_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Combine KAMA crossover direction with Volume Efficiency filter."""
        fast = self._fast_kama[bar_index]
        slow = self._slow_kama[bar_index]
        eff_z = self._vol_eff_z[bar_index]

        if np.isnan(fast) or np.isnan(slow) or np.isnan(eff_z):
            return 0.0

        direction = 1.0 if fast > slow else -1.0

        # Full signal when volume efficiency confirms institutional conviction
        if eff_z > self.vol_eff_threshold:
            return direction

        # Half signal when price is moving but without efficiency
        return direction * 0.5

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "kama",
                "params": {
                    "fast_period": self.kama_fast_period,
                    "slow_period": self.kama_slow_period,
                },
            },
            {
                "name": "volume_efficiency",
                "params": {
                    "period": self.vol_eff_period,
                    "threshold": self.vol_eff_threshold,
                },
            },
        ]
