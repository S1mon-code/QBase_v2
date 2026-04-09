"""MildTrendLongI2hV11 — Keltner Channel (EMA + ATR) + OBV Direction.

Economic logic: Keltner Channels define volatility-adjusted envelopes around an
EMA midline. A close beyond the upper/lower band signals strong directional
momentum. OBV direction confirms that volume is flowing in the same direction,
filtering out low-conviction breakouts that lack volume participation.
"""

from __future__ import annotations

import numpy as np

from indicators._utils import _ema
from indicators.volatility.atr import atr
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV11(TrendingStrategy):
    """Keltner Channel breakout confirmed by OBV direction.

    Signal logic:
        - Close > upper channel AND OBV > OBV_EMA → +1.0
        - Close < lower channel AND OBV < OBV_EMA → -1.0
        - Close inside channel → 0.0

    Attributes:
        ema_period:      EMA midline period.
        atr_period:      ATR period for channel width.
        kelt_mult:       ATR multiplier for channel bands.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_2h_v11"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 64  # max(ema_period, atr_period) + 30

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
        """Precompute Keltner Channel and OBV arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        mid = _ema(self._closes, self.ema_period)
        band = atr(self._highs, self._lows, self._closes, period=self.atr_period) * self.kelt_mult
        self._upper = mid + band
        self._lower = mid - band
        self._obv = obv(self._closes, self._volumes)
        self._obv_ema = _ema(self._obv, 20)

    def _generate_signal(self, bar_index: int) -> float:
        """Return Keltner breakout signal gated by OBV direction."""
        close_val = self._closes[bar_index]
        upper_val = self._upper[bar_index]
        lower_val = self._lower[bar_index]
        obv_val = self._obv[bar_index]
        obv_ema_val = self._obv_ema[bar_index]

        if (
            np.isnan(close_val)
            or np.isnan(upper_val)
            or np.isnan(lower_val)
            or np.isnan(obv_val)
            or np.isnan(obv_ema_val)
        ):
            return 0.0

        obv_rising = obv_val > obv_ema_val
        obv_falling = obv_val < obv_ema_val

        if close_val > upper_val and obv_rising:
            return 1.0
        if close_val < lower_val and obv_falling:
            return -1.0
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "keltner_channel",
                "params": {
                    "ema_period": self.ema_period,
                    "atr_period": self.atr_period,
                    "kelt_mult": self.kelt_mult,
                },
            },
            {"name": "obv", "params": {"ema_period": 20}},
        ]
