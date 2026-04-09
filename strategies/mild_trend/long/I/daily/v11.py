"""MildTrendLongIDailyV11 — MACD Line Direction + CMF Confirmation.

Economic logic: MACD line (fast EMA minus slow EMA) measures intermediate-term
trend direction. Chaikin Money Flow confirms that volume is flowing in the same
direction as price. Agreement of both signals produces a strong trend signal.
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV11(TrendingStrategy):
    """MACD line direction confirmed by Chaikin Money Flow.

    Signal logic:
        - MACD line > 0 AND CMF > 0: long signal (strength = min(1.0, cmf + 0.5))
        - MACD line < 0 AND CMF < 0: short signal (strength = max(-1.0, cmf - 0.5))
        - Disagreement: 0.0 (no signal)

    Attributes:
        fast_period:     MACD fast EMA period.
        slow_period:     MACD slow EMA period.
        signal_period:   MACD signal line period.
        cmf_period:      Chaikin Money Flow lookback period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v11"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 75  # slow_period(26) + signal_period(9) + cmf_period(20) + 20

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    cmf_period: int = 20
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
        """Precompute MACD line and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._macd_line, _, _ = macd(
            self._closes,
            fast=self.fast_period,
            slow=self.slow_period,
            signal=self.signal_period,
        )
        self._cmf = cmf(
            self._highs,
            self._lows,
            self._closes,
            self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on MACD line and CMF agreement."""
        macd_val = self._macd_line[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(macd_val) or np.isnan(cmf_val):
            return 0.0

        if macd_val > 0 and cmf_val > 0:
            return min(1.0, cmf_val + 0.5)
        if macd_val < 0 and cmf_val < 0:
            return max(-1.0, cmf_val - 0.5)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "macd",
                "params": {
                    "fast": self.fast_period,
                    "slow": self.slow_period,
                    "signal": self.signal_period,
                },
            },
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"MACD({self.fast_period},{self.slow_period})",
                [self._make_subplot_trace("MACD Line", datetimes, self._macd_line, color="#bb86fc")],
                zero_line=True,
            ),
            self._make_subplot(
                f"CMF({self.cmf_period})",
                [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
