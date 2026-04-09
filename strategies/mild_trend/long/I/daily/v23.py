"""MildTrendLongIDailyV23 — SuperTrend + CMF Momentum Grading.

Economic logic: SuperTrend provides a clean binary trend direction signal based
on ATR-adaptive bands. Chaikin Money Flow quantifies the buying or selling
pressure behind the move. Grading the signal strength by CMF level produces
a nuanced position sizing signal — full conviction when both trend and money
flow agree strongly, reduced conviction when money flow is weak or absent.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV23(TrendingStrategy):
    """SuperTrend direction graded by Chaikin Money Flow.

    Signal logic (long side; mirror for short):
        - ST direction == 1 AND CMF > 0.1:  +1.0 (strong bullish confirmation)
        - ST direction == 1 AND CMF > 0:    +0.6 (moderate bullish confirmation)
        - ST direction == 1 AND CMF <= 0:   +0.3 (trend only, no volume support)
        - ST direction == -1: mirror negated values

    Attributes:
        st_period:       SuperTrend ATR period.
        st_mult:         SuperTrend ATR multiplier (separate from chandelier_mult).
        cmf_period:      Chaikin Money Flow lookback period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v23"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 54  # st_period(14) + cmf_period(20) + 20

    st_period: int = 14
    st_mult: float = 2.5
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
        """Precompute SuperTrend direction and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_line, self._st_direction = supertrend(
            self._highs,
            self._lows,
            self._closes,
            period=self.st_period,
            multiplier=self.st_mult,
        )
        self._cmf = cmf(
            self._highs,
            self._lows,
            self._closes,
            self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return graded signal based on SuperTrend direction and CMF level."""
        direction = self._st_direction[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(direction) or np.isnan(cmf_val):
            return 0.0

        if direction == 1:
            if cmf_val > 0.1:
                return 1.0
            if cmf_val > 0:
                return 0.6
            return 0.3
        else:
            if cmf_val < -0.1:
                return -1.0
            if cmf_val < 0:
                return -0.6
            return -0.3

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "supertrend",
                "params": {
                    "period": self.st_period,
                    "multiplier": self.st_mult,
                },
            },
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"SuperTrend({self.st_period})", datetimes, self._st_line, style="step", color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"CMF({self.cmf_period})",
                [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
