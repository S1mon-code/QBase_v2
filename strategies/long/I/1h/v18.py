"""ATR Channel Breakout + CMF — Adaptive volatility channel breakout with money-flow confirmation.

An ATR-based channel adapts to the current volatility regime of iron ore.
When price breaks above the upper channel (EMA + ATR multiplier), it indicates
genuine momentum expansion.  Chaikin Money Flow confirms that institutional
money is flowing into the instrument, filtering out false breakouts.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.volatility.atr import atr
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV18(TrendingStrategy):
    """Long signal on ATR channel breakout confirmed by positive CMF.

    Signal logic
    ------------
    * close > ema + atr_mult * atr AND cmf > 0  ->  min(1.0, 0.6 + cmf)
    * close > ema AND cmf > 0.1                 ->  0.4
    * else                                      ->  0.0
    """

    name = "long_I_1h_v18"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 50

    ema_period: int = 30
    atr_period: int = 20
    atr_mult: float = 1.5
    cmf_period: int = 25
    chandelier_mult: float = 2.5

    def on_init_arrays(
        self, closes, highs, lows, opens, volumes, oi, datetimes
    ):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema = ema(closes, self.ema_period)
        self._atr = atr(highs, lows, closes, self.atr_period)
        self._atr_upper = self._ema + self._atr * self.atr_mult
        self._atr_lower = self._ema - self._atr * self.atr_mult
        self._cmf = cmf(highs, lows, closes, volumes, self.cmf_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        ema_val = self._ema[bar_index]
        atr_val = self._atr[bar_index]
        cmf_val = self._cmf[bar_index]

        breakout_level = ema_val + self.atr_mult * atr_val

        if close > breakout_level and cmf_val > 0:
            return min(1.0, 0.6 + cmf_val)
        if close > ema_val and cmf_val > 0.1:
            return 0.4
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"EMA({self.ema_period})", "array": self._ema},
            {"name": "ATR Upper", "array": self._atr_upper, "style": "dash", "type": "overlay"},
            {"name": "ATR Lower", "array": self._atr_lower, "style": "dash", "type": "overlay"},
            {"name": "CMF", "array": self._cmf},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"EMA({self.ema_period})", datetimes, self._ema, color="#ffab40"),
            self._make_overlay(f"ATR Upper({self.atr_period})", datetimes, self._atr_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"ATR Lower({self.atr_period})", datetimes, self._atr_lower, style="dash", color="#26a69a")
        ]
        subplots = [
            self._make_subplot(
                f"CMF({self.cmf_period})",
                [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
