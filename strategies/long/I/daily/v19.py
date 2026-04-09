"""MildTrendLongIDailyV19 — ROC Momentum + OBV EMA Trend.

Economic logic: Rate of Change measures raw price momentum over a fixed
lookback — positive ROC indicates persistent buying pressure. OBV EMA
smooths cumulative volume flow; a rising OBV EMA confirms that volume
is trending with price rather than diverging, separating momentum
continuation from exhaustion moves.
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.roc import rate_of_change
from indicators.volume.obv import obv
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV19(TrendingStrategy):
    """ROC momentum + OBV EMA trend confirmation (Medium Horizon).

    Signal logic:
        - ROC > 0 AND OBV EMA rising  →  positive signal scaled by tanh(ROC/5)
        - ROC < 0 AND OBV EMA falling  →  negative signal scaled by tanh(|ROC|/5)
        - OBV does not confirm ROC direction  →  0.0

    Attributes:
        roc_period:      Lookback for Rate of Change.
        obv_ema_period:  EMA smoothing period applied to OBV.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v19"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    roc_period: int = 20
    obv_ema_period: int = 30
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
        """Precompute ROC and OBV EMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._roc = rate_of_change(self._closes, period=self.roc_period)
        raw_obv = obv(self._closes, self._volumes)
        self._obv_ema = ema(raw_obv, period=self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on ROC and OBV EMA trend."""
        roc_val = self._roc[bar_index]
        obv_ema_curr = self._obv_ema[bar_index]
        obv_ema_prev = self._obv_ema[bar_index - 1] if bar_index > 0 else obv_ema_curr

        if np.isnan(roc_val) or np.isnan(obv_ema_curr) or np.isnan(obv_ema_prev):
            return 0.0

        roc_sign = 1.0 if roc_val > 0 else -1.0 if roc_val < 0 else 0.0

        if roc_sign == 0.0:
            return 0.0

        obv_slope = obv_ema_curr - obv_ema_prev
        obv_ok = (roc_sign > 0 and obv_slope > 0) or (roc_sign < 0 and obv_slope < 0)

        strength = min(1.0, abs(roc_val) / 5.0)

        return roc_sign * strength if obv_ok else 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "rate_of_change", "params": {"period": self.roc_period}},
            {"name": "obv_ema", "params": {"period": self.obv_ema_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"OBV EMA({self.obv_ema_period})",
                [self._make_subplot_trace("OBV EMA", datetimes, self._obv_ema, color="#bb86fc")],
            ),
            self._make_subplot(
                f"ROC({self.roc_period})",
                [self._make_subplot_trace("ROC", datetimes, self._roc, color="#4fc3f7")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
