"""MildTrendLongIDailyV20 — Donchian Breakout + CMF + OBV Triple Confirmation.

Economic logic: Donchian channel breakouts identify price escaping its
recent range — a classical momentum entry signal. Chaikin Money Flow
validates that capital flow supports the breakout. OBV EMA slope
confirms sustained volume accumulation/distribution. Triple agreement
filters out false breakouts driven by thin liquidity or news spikes.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.volume.cmf import cmf
from indicators.volume.obv import obv
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV20(TrendingStrategy):
    """Donchian breakout + CMF + OBV triple confirmation (Medium Horizon).

    Signal logic:
        - Close > upper AND CMF > 0 AND OBV EMA rising  →  +1.0
        - Close < lower AND CMF < 0 AND OBV EMA falling  →  -1.0
        - Breakout with partial confirmation (one of CMF/OBV)  →  ±0.6
        - Inside channel or no confirmation  →  0.0

    Attributes:
        dc_period:       Donchian channel lookback period.
        cmf_period:      Chaikin Money Flow period.
        obv_period:      EMA smoothing period applied to OBV.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_daily_v20"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 80

    dc_period: int = 40
    cmf_period: int = 20
    obv_period: int = 20
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
        """Precompute Donchian, CMF, and OBV EMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dc_upper, self._dc_lower, _dc_mid = donchian(
            self._highs, self._lows, period=self.dc_period
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes, period=self.cmf_period
        )
        raw_obv = obv(self._closes, self._volumes)
        self._obv_ema = ema(raw_obv, period=self.obv_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on Donchian breakout with CMF and OBV."""
        upper = self._dc_upper[bar_index]
        lower = self._dc_lower[bar_index]
        close = self._closes[bar_index]
        cmf_val = self._cmf[bar_index]
        obv_ema_curr = self._obv_ema[bar_index]
        obv_ema_prev = self._obv_ema[bar_index - 1] if bar_index > 0 else obv_ema_curr

        if (
            np.isnan(upper)
            or np.isnan(lower)
            or np.isnan(close)
            or np.isnan(cmf_val)
            or np.isnan(obv_ema_curr)
            or np.isnan(obv_ema_prev)
        ):
            return 0.0

        obv_slope = obv_ema_curr - obv_ema_prev

        if close > upper and cmf_val > 0 and obv_slope > 0:
            return 1.0
        elif close < lower and cmf_val < 0 and obv_slope < 0:
            return -1.0
        elif close > upper and (cmf_val > 0 or obv_slope > 0):
            return 0.6
        elif close < lower and (cmf_val < 0 or obv_slope < 0):
            return -0.6

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "donchian", "params": {"period": self.dc_period}},
            {"name": "cmf", "params": {"period": self.cmf_period}},
            {"name": "obv_ema", "params": {"period": self.obv_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower, style="dash", color="#26a69a")
        ]
        subplots = [
            self._make_subplot(
                f"CMF({self.cmf_period})",
                [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#bb86fc")],
                zero_line=True,
            ),
            self._make_subplot(
                f"OBV EMA({self.obv_period})",
                [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#4fc3f7")],
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
