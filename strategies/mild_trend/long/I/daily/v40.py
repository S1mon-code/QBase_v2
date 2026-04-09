"""MildTrendLongIDailyV40 — KAMA Slope + MACD Direction + CMF (3-Way Confirmation).

Economic logic: KAMA (Kaufman Adaptive Moving Average) adjusts its speed to
the current efficiency ratio, staying flat in sideways markets and tracking
quickly during trends — its slope is a clean trend-existence signal. MACD
line provides momentum direction. Chaikin Money Flow adds volume participation
confirmation. Requiring all three to agree produces the highest-conviction
signal; two-of-three and one-of-three supply graded intermediate signals.
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.trend.kama import kama
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV40(TrendingStrategy):
    """Three-way confirmation: KAMA slope, MACD line direction, and CMF.

    Signal logic (bull_count = number of bullish indicators out of 3):
        - bull_count == 3:  +1.0
        - bull_count == 2:  +0.6
        - bull_count == 1:  +0.2  (implicit: bear_count == 2)
        - bear_count == 2:  -0.6
        - bear_count == 3:  -1.0
        (bull_count and bear_count always sum to 3)

    Attributes:
        kama_period:     KAMA efficiency-ratio lookback period.
        macd_fast:       MACD fast EMA period.
        macd_slow:       MACD slow EMA period.
        cmf_period:      Chaikin Money Flow lookback period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v40"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 66  # macd_slow(26) + kama_period(20) + cmf_period(20)

    kama_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
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
        """Precompute KAMA, MACD line, and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._macd_line, _, _ = macd(
            self._closes,
            fast=self.macd_fast,
            slow=self.macd_slow,
        )
        self._cmf = cmf(
            self._highs,
            self._lows,
            self._closes,
            self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return graded signal based on how many of 3 indicators agree."""
        k = self._kama[bar_index]
        k_prev = self._kama[bar_index - 1]
        macd_val = self._macd_line[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(k) or np.isnan(k_prev) or np.isnan(macd_val) or np.isnan(cmf_val):
            return 0.0

        kama_up = k > k_prev
        macd_up = macd_val > 0
        cmf_up = cmf_val > 0

        bull_count = sum([kama_up, macd_up, cmf_up])
        bear_count = sum([not kama_up, not macd_up, not cmf_up])

        if bull_count == 3:
            return 1.0
        if bear_count == 3:
            return -1.0
        if bull_count == 2:
            return 0.6
        if bear_count == 2:
            return -0.6
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "kama", "params": {"period": self.kama_period}},
            {
                "name": "macd",
                "params": {
                    "fast": self.macd_fast,
                    "slow": self.macd_slow,
                },
            },
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"KAMA({self.kama_period})", datetimes, self._kama, color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"MACD({self.macd_fast},{self.macd_slow})",
                [self._make_subplot_trace("MACD Line", datetimes, self._macd_line, color="#bb86fc")],
                zero_line=True,
            ),
            self._make_subplot(
                f"CMF({self.cmf_period})",
                [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
