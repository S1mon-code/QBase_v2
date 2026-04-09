"""MildTrendLongI4hV15 — EMA(20/60) Cross + OBV Direction.

Economic logic: The 20/60 EMA crossover captures medium-term momentum shifts that
align with the 4h iron ore cycle.  On-Balance Volume trending above its own EMA
confirms that volume is flowing in the direction of the price move.  EMA crossovers
without OBV confirmation are prone to whipsaws in choppy markets.
"""
from __future__ import annotations

import numpy as np

from indicators._utils import _ema
from indicators.volume.obv import obv
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV15(TrendingStrategy):
    """EMA crossover direction confirmed by OBV trend.

    Signal logic:
        EMA_fast > EMA_slow AND OBV > OBV_EMA → +1.0
        EMA_fast < EMA_slow AND OBV < OBV_EMA → -1.0
        EMA agree but no OBV confirm          → 0.5 * direction
        Else → 0.0
    """

    name = "long_I_4h_v15"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 100  # ema_slow + obv_ema_period + 20

    # Optimizable parameters
    ema_fast: int = 20
    ema_slow: int = 60
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
        """Precompute EMA pair and OBV with its own EMA."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._ema_fast = _ema(self._closes, self.ema_fast)
        self._ema_slow = _ema(self._closes, self.ema_slow)
        self._obv = obv(self._closes, self._volumes)
        self._obv_ema = _ema(self._obv, self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from EMA cross filtered by OBV direction."""
        ef = self._ema_fast[bar_index]
        es = self._ema_slow[bar_index]
        obv_val = self._obv[bar_index]
        obv_ema = self._obv_ema[bar_index]

        if np.isnan(ef) or np.isnan(es) or np.isnan(obv_val) or np.isnan(obv_ema):
            return 0.0

        ema_bull = ef > es
        ema_bear = ef < es
        obv_bull = obv_val > obv_ema
        obv_bear = obv_val < obv_ema

        if ema_bull and obv_bull:
            return 1.0
        if ema_bear and obv_bear:
            return -1.0
        if ema_bull:
            return 0.5
        if ema_bear:
            return -0.5

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "ema", "params": {"fast": self.ema_fast, "slow": self.ema_slow}},
            {"name": "obv", "params": {"ema_period": self.obv_ema_period}},
        ]
