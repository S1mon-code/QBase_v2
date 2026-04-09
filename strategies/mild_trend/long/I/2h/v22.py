"""MildTrendLongI2hV22 — DPO + Force Index.

Economic logic: Detrended Price Oscillator isolates the intermediate cycle
component while Force Index combines price change with volume to measure
buying/selling pressure. ATR normalizes DPO for comparable strength across
varying volatility regimes in iron ore.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.dpo import dpo
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV22(TrendingStrategy):
    """DPO cycle direction confirmed by Force Index.

    Signal logic:
        - DPO > 0 AND FI > 0: +min(1.0, DPO / ATR)
        - DPO < 0 AND FI < 0: -min(1.0, abs(DPO) / ATR)
        - Disagree: 0.0
    """

    name = "mild_trend_long_I_2h_v22"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 53  # dpo_period(20) + fi_period(13) + 20

    dpo_period: int = 20
    fi_period: int = 13
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
        """Precompute DPO, Force Index, and ATR arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._dpo = dpo(self._closes, period=self.dpo_period)
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)
        self._atr = atr(self._highs, self._lows, self._closes, period=14)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on DPO and Force Index."""
        d = self._dpo[bar_index]
        f = self._fi[bar_index]
        a = self._atr[bar_index]

        if np.isnan(d) or np.isnan(f) or np.isnan(a) or a == 0.0:
            return 0.0

        strength = float(np.clip(abs(d) / a, 0.0, 1.0))

        if d > 0 and f > 0:
            return strength
        if d < 0 and f < 0:
            return -strength
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "dpo", "params": {"period": self.dpo_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
            {"name": "atr", "params": {"period": 14}},
        ]
