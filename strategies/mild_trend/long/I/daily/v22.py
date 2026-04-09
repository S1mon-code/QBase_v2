"""MildTrendLongIDailyV22 — EMA Crossover + Force Index Confirmation.

Economic logic: EMA crossover (20/60) identifies medium-term trend direction.
Alexander Elder's Force Index measures the power behind each price move by
combining price change with volume. Force Index confirming the EMA direction
validates that institutional money is driving the trend.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV22(TrendingStrategy):
    """EMA crossover with Force Index volume confirmation.

    Signal logic:
        - fast_ema > slow_ema AND Force Index > 0: +1.0 (full long)
        - fast_ema < slow_ema AND Force Index < 0: -1.0 (full short)
        - Disagreement: 0.5 * ema_sign (directional but low conviction)

    Attributes:
        fast_period:     Fast EMA period.
        slow_period:     Slow EMA period.
        fi_period:       Force Index EMA smoothing period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v22"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 93  # slow_period(60) + fi_period(13) + 20

    fast_period: int = 20
    slow_period: int = 60
    fi_period: int = 13
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
        """Precompute fast EMA, slow EMA, and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._fast_ema = ema(self._closes, self.fast_period)
        self._slow_ema = ema(self._closes, self.slow_period)
        self._force_index = force_index(
            self._closes,
            self._volumes,
            period=self.fi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal based on EMA crossover and Force Index agreement."""
        fast = self._fast_ema[bar_index]
        slow = self._slow_ema[bar_index]
        fi_val = self._force_index[bar_index]

        if np.isnan(fast) or np.isnan(slow) or np.isnan(fi_val):
            return 0.0

        ema_sign = 1.0 if fast > slow else -1.0
        fi_confirms = (ema_sign > 0 and fi_val > 0) or (ema_sign < 0 and fi_val < 0)

        return ema_sign if fi_confirms else 0.5 * ema_sign

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "ema_cross",
                "params": {
                    "fast_period": self.fast_period,
                    "slow_period": self.slow_period,
                },
            },
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"EMA({self.fast_period})", datetimes, self._fast_ema, color="#ffab40"),
            self._make_overlay(f"EMA({self.slow_period})", datetimes, self._slow_ema, color="#ab47bc")
        ]
        subplots = [
            self._make_subplot(
                f"Force Index({self.fi_period})",
                [self._make_subplot_trace("Force Index", datetimes, self._force_index, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
