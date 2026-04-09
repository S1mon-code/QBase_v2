"""MildTrendLongI4hV11 — Donchian(40) + Force Index.

Economic logic: Donchian breakouts capture range expansion at key price levels.
Force Index confirms that volume-weighted momentum supports the breakout direction.
Breakouts without volume conviction are more likely to be false signals that trap
momentum chasers.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV11(TrendingStrategy):
    """Donchian breakout signals confirmed by Force Index momentum.

    Signal logic:
        Close > upper AND FI > 0 → +1.0
        Close < lower AND FI < 0 → -1.0
        Close > mid   AND FI > 0 → +0.5
        Close < mid   AND FI < 0 → -0.5
        Else → 0.0
    """

    name = "long_I_4h_v11"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 73  # donch_period + fi_period + 20

    # Optimizable parameters
    donch_period: int = 40
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
        """Precompute Donchian channels and Force Index."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._donch_upper, self._donch_mid, self._donch_lower = donchian(
            self._highs, self._lows, period=self.donch_period,
        )
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from Donchian breakout filtered by Force Index."""
        c = self._closes[bar_index]
        upper = self._donch_upper[bar_index]
        mid = self._donch_mid[bar_index]
        lower = self._donch_lower[bar_index]
        fi_val = self._fi[bar_index]

        if np.isnan(c) or np.isnan(upper) or np.isnan(fi_val):
            return 0.0

        if c > upper and fi_val > 0.0:
            return 1.0
        if c < lower and fi_val < 0.0:
            return -1.0
        if c > mid and fi_val > 0.0:
            return 0.5
        if c < mid and fi_val < 0.0:
            return -0.5

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "donchian", "params": {"period": self.donch_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]
