"""MildTrendLongI2hV25 — HMA Crossover + OI Flow.

Economic logic: Hull Moving Average provides responsive trend detection with
minimal lag via its weighted moving average construction. OI Flow combines
open interest changes with volume and price direction to confirm whether
institutional positioning supports the trend in iron ore futures.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.hma import hma
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV25(TrendingStrategy):
    """HMA fast/slow crossover confirmed by OI Flow.

    Signal logic:
        - HMA_fast > HMA_slow AND flow > signal: +1.0
        - HMA_fast < HMA_slow AND flow < signal: -1.0
        - HMA agrees but no OI confirm: 0.4 * direction
        - Disagree: 0.0
    """

    name = "long_I_2h_v25"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["trend", "volume"]
    warmup: int = 84  # hma_slow(50) + oi_period(14) + 20

    hma_fast: int = 20
    hma_slow: int = 50
    oi_period: int = 14
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
        """Precompute HMA fast, HMA slow, and OI Flow arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._hma_fast = hma(self._closes, period=self.hma_fast)
        self._hma_slow = hma(self._closes, period=self.hma_slow)
        flow, sig = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oi_period,
        )
        self._oi_flow = flow
        self._oi_sig = sig

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on HMA crossover and OI Flow."""
        hf = self._hma_fast[bar_index]
        hs = self._hma_slow[bar_index]
        fl = self._oi_flow[bar_index]
        sg = self._oi_sig[bar_index]

        if np.isnan(hf) or np.isnan(hs) or np.isnan(fl) or np.isnan(sg):
            return 0.0

        if hf > hs:
            return 1.0 if fl > sg else 0.4
        if hf < hs:
            return -1.0 if fl < sg else -0.4
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "hma_fast", "params": {"period": self.hma_fast}},
            {"name": "hma_slow", "params": {"period": self.hma_slow}},
            {"name": "oi_flow", "params": {"period": self.oi_period}},
        ]
