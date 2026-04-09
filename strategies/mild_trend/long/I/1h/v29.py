"""MildTrendLongI1hV29 — True Strength Index + Volume Profile POC.

Economic logic: The True Strength Index (TSI) double-smooths momentum to
produce a clean oscillator that captures trend direction and strength without
whipsaw noise. A positive TSI above its signal line confirms sustained
bullish momentum. Volume Profile's Point of Control (POC) identifies the
price level with the highest traded volume — when price is above POC, the
market is trading above the most-accepted value area, indicating bullish
price discovery. Together, TSI momentum + price above POC value creates a
confluence of technical momentum and market microstructure support.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.tsi import tsi
from indicators.volume.volume_profile import poc
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV29(TrendingStrategy):
    """TSI bullish momentum + price above POC confirmation.

    Signal logic:
        TSI > signal AND close > POC -> strong (0.7-1.0)
        TSI > 0 AND close > POC      -> moderate (0.4-0.6)
        TSI > 0 only                 -> weak (0.2-0.3)
        else                         -> 0.0
    """

    name = "mild_trend_long_I_1h_v29"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 55  # tsi long_period(25) + short_period(13) + signal + buffer

    # Optimizable parameters
    tsi_long: int = 25
    tsi_short: int = 13
    poc_period: int = 30
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
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=self.tsi_short,
        )
        self._poc = poc(
            self._closes, self._volumes, period=self.poc_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        tsi_val = self._tsi_line[bar_index]
        tsi_sig = self._tsi_signal[bar_index]
        poc_val = self._poc[bar_index]
        close_val = self._closes[bar_index]

        if np.isnan(tsi_val):
            return 0.0

        # TSI must be positive for long signals
        if tsi_val <= 0.0:
            return 0.0

        # TSI strength (roughly -100 to 100 range)
        tsi_strength = min(1.0, tsi_val / 30.0)

        # Check if TSI is above signal line
        tsi_above_signal = not np.isnan(tsi_sig) and tsi_val > tsi_sig

        # Check if price is above POC
        price_above_poc = not np.isnan(poc_val) and close_val > poc_val

        if tsi_above_signal and price_above_poc:
            # Full confirmation: TSI momentum + price in value
            poc_distance = (close_val - poc_val) / poc_val
            return min(1.0, 0.6 + tsi_strength * 0.2 + poc_distance * 5.0)

        if tsi_above_signal:
            return min(0.5, 0.3 + tsi_strength * 0.2)

        if price_above_poc:
            return min(0.4, 0.2 + tsi_strength * 0.2)

        # TSI positive but weak
        return min(0.3, 0.2 + tsi_strength * 0.1)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"POC({self.poc_period})", "array": self._poc, "style": "step"},
            {"name": "TSI Line", "array": self._tsi_line, "panel": "TSI"},
            {"name": "TSI Signal", "array": self._tsi_signal, "panel": "TSI"},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"TSI({self.tsi_long},{self.tsi_short})",
                [
                    self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._tsi_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            ),
            self._make_subplot(
                f"POC({self.poc_period})",
                [self._make_subplot_trace("POC", datetimes, self._poc, color="#bb86fc")],
            )
        ]
        return {"overlays": [], "subplots": subplots}
