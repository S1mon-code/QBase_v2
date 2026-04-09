"""MildTrendLongIDailyV12 — EMA Crossover + OBV Slope Confirmation.

Economic logic: EMA crossover identifies medium-term trend direction. OBV slope
(measured as the derivative of a smoothed OBV EMA) confirms that volume is
accumulating in the trend direction. Agreement produces a full signal; disagreement
yields a weakened 0.5x signal to stay positioned but with lower conviction.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV12(TrendingStrategy):
    """EMA crossover with OBV slope confirmation.

    Signal logic:
        - fast_ema > slow_ema AND OBV EMA slope > 0: +1.0 (full long)
        - fast_ema < slow_ema AND OBV EMA slope < 0: -1.0 (full short)
        - EMA gives direction but OBV disagrees: 0.5 * ema_sign (weak signal)

    Attributes:
        fast_period:     Fast EMA period.
        slow_period:     Slow EMA period.
        obv_period:      EMA smoothing period applied to raw OBV.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v12"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 85  # slow_period(60) + obv_period(20) + 5

    fast_period: int = 20
    slow_period: int = 60
    obv_period: int = 20
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
        """Precompute fast EMA, slow EMA, and OBV EMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._fast_ema = ema(self._closes, self.fast_period)
        self._slow_ema = ema(self._closes, self.slow_period)
        obv_raw = obv(self._closes, self._volumes)
        self._obv_ema = ema(obv_raw, self.obv_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal based on EMA crossover and OBV slope agreement."""
        fast = self._fast_ema[bar_index]
        slow = self._slow_ema[bar_index]
        obv_ema_now = self._obv_ema[bar_index]

        if np.isnan(fast) or np.isnan(slow) or np.isnan(obv_ema_now):
            return 0.0

        obv_slope = 0.0
        if bar_index > 0 and not np.isnan(self._obv_ema[bar_index - 1]):
            obv_slope = obv_ema_now - self._obv_ema[bar_index - 1]

        ema_sign = 1.0 if fast > slow else -1.0
        obv_confirms = (ema_sign > 0 and obv_slope > 0) or (
            ema_sign < 0 and obv_slope < 0
        )

        return ema_sign if obv_confirms else 0.5 * ema_sign

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
            {"name": "obv_ema", "params": {"obv_period": self.obv_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"EMA({self.fast_period})", datetimes, self._fast_ema, color="#ffab40"),
            self._make_overlay(f"EMA({self.slow_period})", datetimes, self._slow_ema, color="#ab47bc")
        ]
        subplots = [
            self._make_subplot(
                f"OBV EMA({self.obv_period})",
                [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#bb86fc")],
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
