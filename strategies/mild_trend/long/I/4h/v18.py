"""MildTrendLongI4hV18 — KAMA Slope + TRIX Direction.

Economic logic: KAMA (Kaufman Adaptive Moving Average) adapts its smoothing to
market noise — fast in trends, slow in chop.  TRIX is a triple-smoothed EMA
rate-of-change that filters out short-term noise.  When both adaptive indicators
agree on direction, the trend signal is highly reliable for iron ore's medium-term
cycles.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.trix import trix
from indicators.trend.kama import kama
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV18(TrendingStrategy):
    """KAMA slope direction confirmed by TRIX crossover.

    Signal logic:
        KAMA rising AND TRIX line > TRIX signal → +1.0
        KAMA falling AND TRIX < signal          → -1.0
        Disagree → 0.0
    """

    name = "mild_trend_long_I_4h_v18"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum"]
    warmup: int = 85  # kama_period + trix_period * 3 + 20

    # Optimizable parameters
    kama_period: int = 20
    trix_period: int = 15
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
        """Precompute KAMA and TRIX arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kama = kama(self._closes, period=self.kama_period)
        self._trix_line, self._trix_signal = trix(self._closes, period=self.trix_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from KAMA slope direction filtered by TRIX."""
        if bar_index < 1:
            return 0.0

        kama_cur = self._kama[bar_index]
        kama_prev = self._kama[bar_index - 1]
        trix_val = self._trix_line[bar_index]
        trix_sig = self._trix_signal[bar_index]

        if np.isnan(kama_cur) or np.isnan(kama_prev) or np.isnan(trix_val) or np.isnan(trix_sig):
            return 0.0

        kama_rising = kama_cur > kama_prev
        kama_falling = kama_cur < kama_prev
        trix_bull = trix_val > trix_sig
        trix_bear = trix_val < trix_sig

        if kama_rising and trix_bull:
            return 1.0
        if kama_falling and trix_bear:
            return -1.0

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "kama", "params": {"period": self.kama_period}},
            {"name": "trix", "params": {"period": self.trix_period}},
        ]
