"""MildTrendLongI4hV20 — Vortex Indicator + OBV Direction.

Economic logic: The Vortex Indicator captures positive and negative trend movement
via true range calculations.  OBV provides cumulative volume confirmation — when
volume consistently flows with the Vortex direction, the trend has broad market
participation.  The VI+/VI- differential scales signal strength, giving larger
signals to stronger Vortex readings.
"""
from __future__ import annotations

import numpy as np

from indicators._utils import _ema
from indicators.trend.vortex import vortex
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV20(TrendingStrategy):
    """Vortex differential direction confirmed by OBV trend.

    Signal logic:
        VI+ > VI- AND OBV > OBV_EMA → +(VI+ - VI-) clipped [0, 1]
        VI- > VI+ AND OBV < OBV_EMA → -(VI- - VI+) clipped [-1, 0]
        Disagree → 0.0
    """

    name = "mild_trend_long_I_4h_v20"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 54  # vortex_period + obv_ema_period + 20

    # Optimizable parameters
    vortex_period: int = 14
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
        """Precompute Vortex and OBV with EMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period,
        )
        self._obv = obv(self._closes, self._volumes)
        self._obv_ema = _ema(self._obv, self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from Vortex differential filtered by OBV direction."""
        vi_p = self._vi_plus[bar_index]
        vi_m = self._vi_minus[bar_index]
        obv_val = self._obv[bar_index]
        obv_ema = self._obv_ema[bar_index]

        if np.isnan(vi_p) or np.isnan(vi_m) or np.isnan(obv_val) or np.isnan(obv_ema):
            return 0.0

        obv_bull = obv_val > obv_ema
        obv_bear = obv_val < obv_ema

        if vi_p > vi_m and obv_bull:
            return min(1.0, max(0.0, vi_p - vi_m))
        if vi_m > vi_p and obv_bear:
            return max(-1.0, min(0.0, -(vi_m - vi_p)))

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "vortex", "params": {"period": self.vortex_period}},
            {"name": "obv", "params": {"ema_period": self.obv_ema_period}},
        ]
