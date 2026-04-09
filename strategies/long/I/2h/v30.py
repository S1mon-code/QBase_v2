"""MildTrendLongI2hV30 — KST + Force Index.

Economic logic: Know Sure Thing oscillator combines four ROC periods with
smoothing to capture multi-timeframe momentum consensus. Force Index measures
buying/selling pressure through price change weighted by volume. Agreement
between KST momentum direction and volume-based force confirms sustainable
iron ore trend moves on the 2h timeframe.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.kst import kst
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV30(TrendingStrategy):
    """KST line vs signal crossover confirmed by Force Index.

    Signal logic:
        - KST > signal AND FI > 0: +min(1.0, abs(kst)/10)
        - KST < signal AND FI < 0: -min(1.0, abs(kst)/10)
        - Disagree: 0.0
    """

    name = "long_I_2h_v30"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 83  # 50 + fi_period(13) + 20

    kst_signal_period: int = 9
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
        """Precompute KST and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        kst_line, signal_line = kst(self._closes)
        self._kst = kst_line
        self._kst_sig = signal_line
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on KST and Force Index."""
        k = self._kst[bar_index]
        ks = self._kst_sig[bar_index]
        f = self._fi[bar_index]

        if np.isnan(k) or np.isnan(ks) or np.isnan(f):
            return 0.0

        strength = float(np.clip(abs(k) / 10.0, 0.0, 1.0))

        if k > ks and f > 0:
            return strength
        if k < ks and f < 0:
            return -strength
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "kst", "params": {"signal_period": self.kst_signal_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]
