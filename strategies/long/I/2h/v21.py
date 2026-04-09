"""MildTrendLongI2hV21 — CMO + OBV EMA Direction.

Economic logic: Chande Momentum Oscillator measures raw momentum strength
while OBV relative to its EMA confirms whether volume flow supports the
price trend. Agreement between momentum direction and volume accumulation
produces high-conviction signals for iron ore's 2h timeframe.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cmo import cmo
from indicators.volume.obv import obv
from indicators._utils import _ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV21(TrendingStrategy):
    """CMO direction confirmed by OBV vs its EMA.

    Signal logic:
        - CMO > 0 AND OBV > OBV_EMA: +min(1.0, abs(CMO)/100)
        - CMO < 0 AND OBV < OBV_EMA: -min(1.0, abs(CMO)/100)
        - Disagree: 0.0
    """

    name = "long_I_2h_v21"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 54  # cmo_period(14) + obv_ema_period(20) + 20

    cmo_period: int = 14
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
        """Precompute CMO, OBV, and OBV EMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._cmo = cmo(self._closes, period=self.cmo_period)
        obv_arr = obv(self._closes, self._volumes)
        self._obv = obv_arr
        self._obv_ema = _ema(obv_arr, self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on CMO and OBV direction."""
        c = self._cmo[bar_index]
        o = self._obv[bar_index]
        oe = self._obv_ema[bar_index]

        if np.isnan(c) or np.isnan(o) or np.isnan(oe):
            return 0.0

        strength = float(np.clip(abs(c) / 100.0, 0.0, 1.0))

        if c > 0 and o > oe:
            return strength
        if c < 0 and o < oe:
            return -strength
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "cmo", "params": {"period": self.cmo_period}},
            {"name": "obv", "params": {}},
            {"name": "obv_ema", "params": {"period": self.obv_ema_period}},
        ]
