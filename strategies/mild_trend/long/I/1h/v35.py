"""MildTrendLongI1hV35 — Keltner Channel (EMA + ATR) + OBV Direction.

Economic logic: Keltner channels define a volatility-adjusted trend envelope.
Breakouts beyond the channel indicate genuine trend acceleration, not just
noise.  OBV rising above its EMA confirms institutional accumulation is
driving the move.  Range-bound traders who fade channel breakouts without
checking volume get squeezed when both conditions align.
"""
from __future__ import annotations

import numpy as np

from indicators._utils import _ema
from indicators.volatility.atr import atr
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV35(TrendingStrategy):
    """Keltner Channel breakout confirmed by OBV EMA direction.

    Signal logic:
        Close > upper AND OBV > OBV_EMA → +1.0
        Close < lower AND OBV < OBV_EMA → -1.0
        Inside channel → 0.0
    """

    name = "mild_trend_long_I_1h_v35"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 54  # max(ema_period, atr_period) + 20 + buffer

    # Optimizable parameters
    ema_period: int = 20
    atr_period: int = 14
    kelt_mult: float = 1.5
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
        """Precompute Keltner Channel and OBV + OBV_EMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        mid = _ema(self._closes, self.ema_period)
        channel = atr(self._highs, self._lows, self._closes, period=self.atr_period)
        self._kelt_upper = mid + channel * self.kelt_mult
        self._kelt_lower = mid - channel * self.kelt_mult

        raw_obv = obv(self._closes, self._volumes)
        self._obv = raw_obv
        self._obv_ema = _ema(raw_obv, self.ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from Keltner breakout + OBV direction."""
        close = self._closes[bar_index]
        upper = self._kelt_upper[bar_index]
        lower = self._kelt_lower[bar_index]
        obv_val = self._obv[bar_index]
        obv_ema_val = self._obv_ema[bar_index]

        if (
            np.isnan(close) or np.isnan(upper) or np.isnan(lower)
            or np.isnan(obv_val) or np.isnan(obv_ema_val)
        ):
            return 0.0

        obv_rising = obv_val > obv_ema_val
        obv_falling = obv_val < obv_ema_val

        if close > upper and obv_rising:
            return 1.0
        if close < lower and obv_falling:
            return -1.0

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "keltner_channel", "params": {
                "ema_period": self.ema_period,
                "atr_period": self.atr_period,
                "multiplier": self.kelt_mult,
            }},
            {"name": "obv", "params": {}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"Keltner Upper({self.ema_period})", datetimes, self._kelt_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"Keltner Lower({self.ema_period})", datetimes, self._kelt_lower, style="dash", color="#26a69a")
        ]
        subplots = [
            self._make_subplot(
                "OBV EMA",
                [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#bb86fc")],
            )
        ]
        return {"overlays": overlays, "subplots": subplots}

