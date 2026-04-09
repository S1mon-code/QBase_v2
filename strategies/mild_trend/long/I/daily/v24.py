"""MildTrendLongIDailyV24 — ADX Trend Strength + OBV EMA Trend Alignment.

Economic logic: ADX measures trend strength without regard to direction.
Above 25 indicates a trending regime where directional strategies should be
active; below 20 indicates a ranging market. OBV EMA slope and the DI lines
together identify the trend direction and whether volume is accumulating in
that direction. The combination filters out weak, choppy price action.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx_with_di
from indicators.trend.ema import ema
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV24(TrendingStrategy):
    """ADX trend strength filter with OBV EMA and DI directional alignment.

    Signal logic:
        - ADX > 25 AND OBV_EMA rising AND plus_di > minus_di: +1.0
        - ADX > 25 AND OBV_EMA falling AND minus_di > plus_di: -1.0
        - ADX 20–25: half signal (0.5 * directional signal)
        - ADX < 20: 0.0 (no signal, ranging market)

    Attributes:
        adx_period:      ADX / DI calculation period.
        obv_ema_period:  EMA smoothing period for raw OBV.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v24"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 68  # adx_period*2(28) + obv_ema_period(20) + 20

    adx_period: int = 14
    obv_ema_period: int = 20
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
        """Precompute ADX, DI lines, and OBV EMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._adx, self._plus_di, self._minus_di = adx_with_di(
            self._highs,
            self._lows,
            self._closes,
            period=self.adx_period,
        )
        obv_raw = obv(self._closes, self._volumes)
        self._obv_ema = ema(obv_raw, self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal gated by ADX strength and directional alignment."""
        adx_val = self._adx[bar_index]
        plus_di = self._plus_di[bar_index]
        minus_di = self._minus_di[bar_index]
        obv_now = self._obv_ema[bar_index]

        if np.isnan(adx_val) or np.isnan(plus_di) or np.isnan(minus_di) or np.isnan(obv_now):
            return 0.0

        if adx_val < 20:
            return 0.0

        obv_slope = 0.0
        if bar_index > 0 and not np.isnan(self._obv_ema[bar_index - 1]):
            obv_slope = obv_now - self._obv_ema[bar_index - 1]

        if obv_slope > 0 and plus_di > minus_di:
            raw_signal = 1.0
        elif obv_slope < 0 and minus_di > plus_di:
            raw_signal = -1.0
        else:
            return 0.0

        if adx_val >= 25:
            return raw_signal
        return 0.5 * raw_signal

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "adx_with_di",
                "params": {"period": self.adx_period},
            },
            {"name": "obv_ema", "params": {"obv_ema_period": self.obv_ema_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"ADX({self.adx_period})",
                [
                    self._make_subplot_trace("ADX", datetimes, self._adx, color="#bb86fc"),
                    self._make_subplot_trace("+DI", datetimes, self._plus_di, color="#26a69a"),
                    self._make_subplot_trace("-DI", datetimes, self._minus_di, color="#ef5350"),
                ],
            ),
            self._make_subplot(
                f"OBV EMA({self.obv_ema_period})",
                [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#bb86fc")],
            )
        ]
        return {"overlays": [], "subplots": subplots}
