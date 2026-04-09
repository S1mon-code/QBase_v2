"""Bollinger Band Breakout + OBV Trend — Volatility breakout confirmed by volume flow.

When iron ore breaks above the upper Bollinger Band it signals expanding volatility
in the trend direction.  On-balance volume trending above its own EMA confirms that
the breakout is backed by genuine buying pressure rather than a low-volume spike.
"""
from __future__ import annotations

import numpy as np

from indicators.volatility.bollinger import bollinger_bands
from indicators.volume.obv import obv
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV17(TrendingStrategy):
    """Long signal on Bollinger upper-band breakout with OBV confirmation.

    Signal logic
    ------------
    * close > upper_band AND obv > ema(obv)  ->  0.9  (strong breakout)
    * close > middle_band AND obv > ema(obv) ->  0.5  (above middle with volume)
    * else                                   ->  0.0
    """

    name = "mild_trend_long_I_1h_v17"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    bb_period: int = 30
    bb_std: float = 2.0
    obv_ema_period: int = 20
    chandelier_mult: float = 2.5

    def on_init_arrays(
        self, closes, highs, lows, opens, volumes, oi, datetimes
    ):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._bb_upper, self._bb_middle, self._bb_lower = bollinger_bands(
            closes, self.bb_period, self.bb_std
        )
        self._obv = obv(closes, volumes)
        self._obv_ema = ema(self._obv, self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        upper = self._bb_upper[bar_index]
        middle = self._bb_middle[bar_index]
        obv_val = self._obv[bar_index]
        obv_ema_val = self._obv_ema[bar_index]

        if close > upper and obv_val > obv_ema_val:
            return 0.9
        if close > middle and obv_val > obv_ema_val:
            return 0.5
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "Bollinger Upper", "array": self._bb_upper, "style": "dash"},
            {"name": "Bollinger Middle", "array": self._bb_middle},
            {"name": "Bollinger Lower", "array": self._bb_lower, "style": "dash"},
            {"name": "OBV", "array": self._obv, "panel": "OBV"},
            {"name": "OBV EMA", "array": self._obv_ema, "panel": "OBV"},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"BB Upper({self.bb_period})", datetimes, self._bb_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"BB Mid({self.bb_period})", datetimes, self._bb_middle, color="#ffab40"),
            self._make_overlay(f"BB Lower({self.bb_period})", datetimes, self._bb_lower, style="dash", color="#26a69a")
        ]
        subplots = [
            self._make_subplot(
                f"OBV EMA({self.obv_ema_period})",
                [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#bb86fc")],
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
