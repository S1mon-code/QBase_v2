"""MildTrendLongI2hV14 — TRIX + Force Index Confirmation.

Economic logic: TRIX is a triple-smoothed momentum oscillator that filters out
short-term noise, making it ideal for medium-horizon trend detection. Force Index
combines price change with volume, measuring the true force behind moves. When
TRIX confirms trend direction and Force Index shows volume commitment, the
probability of trend continuation is high.
"""

from __future__ import annotations

import numpy as np

from indicators.momentum.trix import trix
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV14(TrendingStrategy):
    """TRIX momentum with Force Index volume confirmation.

    Signal logic:
        - TRIX line > TRIX signal AND FI > 0 → +min(1.0, abs(trix_val) * 100)
        - TRIX line < TRIX signal AND FI < 0 → -min(1.0, abs(trix_val) * 100)
        - Disagree → 0.0

    Attributes:
        trix_period:     TRIX calculation period.
        fi_period:       Force Index smoothing period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_2h_v14"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 78  # trix_period * 3 + fi_period + 20

    trix_period: int = 15
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
        """Precompute TRIX and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        trix_result = trix(self._closes, period=self.trix_period)
        self._trix_line = trix_result[0]
        self._trix_signal = trix_result[1]
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return TRIX directional signal confirmed by Force Index."""
        trix_val = self._trix_line[bar_index]
        trix_sig = self._trix_signal[bar_index]
        fi_val = self._fi[bar_index]

        if np.isnan(trix_val) or np.isnan(trix_sig) or np.isnan(fi_val):
            return 0.0

        if trix_val > trix_sig and fi_val > 0.0:
            return min(1.0, abs(trix_val) * 100.0)
        if trix_val < trix_sig and fi_val < 0.0:
            return -min(1.0, abs(trix_val) * 100.0)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "trix", "params": {"period": self.trix_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]
