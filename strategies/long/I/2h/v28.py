"""MildTrendLongI2hV28 — Schaff Trend Cycle + OBV Direction.

Economic logic: Schaff Trend Cycle applies a double-smoothed stochastic to
MACD, producing a fast-reacting oscillator that identifies trend transitions
earlier. OBV relative to its EMA confirms volume flow supports the detected
trend phase in iron ore's 2h timeframe.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.volume.obv import obv
from indicators._utils import _ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV28(TrendingStrategy):
    """Schaff Trend Cycle zones confirmed by OBV vs EMA.

    Signal logic:
        - STC > 75 AND OBV > OBV_EMA: +min(1.0, STC/100)
        - STC < 25 AND OBV < OBV_EMA: -min(1.0, (100-STC)/100)
        - STC in [25, 75]: 0.0
    """

    name = "long_I_2h_v28"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 113  # stc_fast(23) + obv_ema_period(20) + 50 + 20

    stc_period: int = 10
    stc_fast: int = 23
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
        """Precompute STC, OBV, and OBV EMA arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._stc = schaff_trend_cycle(
            self._closes, period=self.stc_period, fast=self.stc_fast, slow=50,
        )
        obv_arr = obv(self._closes, self._volumes)
        self._obv = obv_arr
        self._obv_ema = _ema(obv_arr, self.obv_ema_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on STC zones and OBV direction."""
        stc = self._stc[bar_index]
        o = self._obv[bar_index]
        oe = self._obv_ema[bar_index]

        if np.isnan(stc) or np.isnan(o) or np.isnan(oe):
            return 0.0

        if stc > 75 and o > oe:
            return float(np.clip(stc / 100.0, 0.0, 1.0))
        if stc < 25 and o < oe:
            return -float(np.clip((100.0 - stc) / 100.0, 0.0, 1.0))
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "schaff_trend_cycle", "params": {"period": self.stc_period, "fast": self.stc_fast, "slow": 50}},
            {"name": "obv", "params": {}},
            {"name": "obv_ema", "params": {"period": self.obv_ema_period}},
        ]
