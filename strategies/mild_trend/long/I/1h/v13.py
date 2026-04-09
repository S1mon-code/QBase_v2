"""MildTrendLongI1hV13 — MACD + CMF (1H tuned).

Economic logic: MACD line above zero confirms medium-term momentum while
Chaikin Money Flow validates that buying pressure dominates accumulation.
When both align, institutional capital is actively pushing prices higher.
Parameters tuned for 1H iron ore: wider MACD windows (30/65/22) capture
multi-day trend momentum, and 40-bar CMF smooths out intraday noise.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV13(TrendingStrategy):
    """MACD line > 0 with positive CMF accumulation.

    Signal logic:
        macd_line > 0 AND cmf > 0 -> signal = min(1.0, cmf + 0.5)
        else -> 0.0
    """

    name = "mild_trend_long_I_1h_v13"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 120  # slow_period(65) + signal_period(22) + cmf_period(40) buffer

    # Optimizable parameters
    fast_period: int = 30
    slow_period: int = 65
    signal_period: int = 22
    cmf_period: int = 40
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
        """Precompute MACD and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            self._closes, fast=self.fast_period, slow=self.slow_period,
            signal=self.signal_period,
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal based on MACD momentum with CMF volume confirmation."""
        macd_val = self._macd_line[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(macd_val) or np.isnan(cmf_val):
            return 0.0

        if macd_val > 0.0 and cmf_val > 0.0:
            return min(1.0, cmf_val + 0.5)

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for auto-generated panels."""
        return [
            {"name": "MACD Line", "array": self._macd_line, "panel": "MACD"},
            {"name": "Signal Line", "array": self._macd_signal, "panel": "MACD"},
            {"name": "Histogram", "array": self._macd_hist, "panel": "MACD", "style": "bar",
             "color_positive": "#26a69a", "color_negative": "#ef5350"},
            {"name": "CMF", "array": self._cmf},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"MACD({self.fast_period},{self.slow_period},{self.signal_period})",
                [
                    self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._macd_signal, color="#4fc3f7"),
                    self._make_subplot_trace("Histogram", datetimes, self._macd_hist, style="bar", color_positive="#26a69a", color_negative="#ef5350"),
                ],
                zero_line=True,
            ),
            self._make_subplot(
                f"CMF({self.cmf_period})",
                [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
