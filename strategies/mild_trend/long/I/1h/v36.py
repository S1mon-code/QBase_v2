"""MildTrendLongI1hV36 — Williams %R + MACD Line Direction.

Economic logic: Williams %R identifies overbought/oversold extremes while MACD
line provides trend direction bias.  When %R crosses out of an extreme zone in
the direction MACD already confirms, the reversal-to-trend move is strongest.
Counter-trend traders who fade these crossovers without checking MACD direction
get caught on the wrong side.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.momentum.williams_r import williams_r
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV36(TrendingStrategy):
    """Williams %R crossover signals filtered by MACD line direction.

    Signal logic:
        MACD > 0 AND %R crosses above -80 → +0.8
        MACD < 0 AND %R crosses below -20 → -0.8
        MACD > 0 AND %R > -50            → +0.4
        MACD < 0 AND %R < -50            → -0.4
        Else → 0.0
    """

    name = "mild_trend_long_I_1h_v36"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 60  # macd_slow + wr_period + 20

    # Optimizable parameters
    wr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
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
        """Precompute Williams %R and MACD arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._wr = williams_r(
            self._highs, self._lows, self._closes, period=self.wr_period,
        )
        self._macd_line, _, _ = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from %R crossovers filtered by MACD direction."""
        wr_val = self._wr[bar_index]
        wr_prev = self._wr[bar_index - 1]
        macd_val = self._macd_line[bar_index]

        if np.isnan(wr_val) or np.isnan(wr_prev) or np.isnan(macd_val):
            return 0.0

        macd_bull = macd_val > 0.0
        macd_bear = macd_val < 0.0

        # Strong signals — crossover at extreme zones
        if macd_bull and wr_val > -80.0 and wr_prev <= -80.0:
            return 0.8
        if macd_bear and wr_val < -20.0 and wr_prev >= -20.0:
            return -0.8

        # Moderate signals — trend continuation
        if macd_bull and wr_val > -50.0:
            return 0.4
        if macd_bear and wr_val < -50.0:
            return -0.4

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "williams_r", "params": {"period": self.wr_period}},
            {"name": "macd", "params": {"fast": self.macd_fast, "slow": self.macd_slow}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"MACD({self.macd_fast},{self.macd_slow})",
                [self._make_subplot_trace("MACD Line", datetimes, self._macd_line, color="#bb86fc")],
                zero_line=True,
            ),
            self._make_subplot(
                f"Williams %R({self.wr_period})",
                [self._make_subplot_trace("Williams %R", datetimes, self._wr, color="#bb86fc")],
                horizontal_lines=[-80, -20], y_range=[-100, 0],
            )
        ]
        return {"overlays": [], "subplots": subplots}

