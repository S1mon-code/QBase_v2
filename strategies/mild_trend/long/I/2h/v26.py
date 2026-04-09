"""MildTrendLongI2hV26 — MACD Histogram + Bollinger Band Position.

Economic logic: MACD histogram direction (rising/falling) captures momentum
acceleration while Bollinger Band position measures where price sits within
the volatility envelope. When momentum accelerates in the direction of the
BB positioning, iron ore trends are likely to persist.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.macd import macd
from indicators.volatility.bollinger import bollinger_bands
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV26(TrendingStrategy):
    """MACD histogram momentum confirmed by Bollinger Band position.

    Signal logic:
        - MACD hist > 0 AND rising AND bb_pos > 0.5: +min(1.0, bb_pos)
        - MACD hist < 0 AND falling AND bb_pos < 0.5: -min(1.0, 1.0-bb_pos)
        - Disagree: 0.0
    """

    name = "mild_trend_long_I_2h_v26"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 46  # max(macd_slow(26), bb_period(20)) + 20

    macd_fast: int = 12
    macd_slow: int = 26
    bb_period: int = 20
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
        """Precompute MACD and Bollinger Bands arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        _, _, hist = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow, signal=9,
        )
        self._hist = hist
        upper, _, lower = bollinger_bands(
            self._closes, period=self.bb_period, num_std=2.0,
        )
        self._bb_upper = upper
        self._bb_lower = lower

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on MACD hist and BB position."""
        h = self._hist[bar_index]
        u = self._bb_upper[bar_index]
        lo = self._bb_lower[bar_index]
        c = self._closes[bar_index]

        if np.isnan(h) or np.isnan(u) or np.isnan(lo) or np.isnan(c):
            return 0.0

        if bar_index < 1:
            return 0.0
        h_prev = self._hist[bar_index - 1]
        if np.isnan(h_prev):
            return 0.0

        bb_range = u - lo
        if bb_range <= 0:
            return 0.0

        bb_pos = (c - lo) / bb_range  # 0 to 1

        if h > 0 and h > h_prev and bb_pos > 0.5:
            return float(np.clip(bb_pos, 0.0, 1.0))
        if h < 0 and h < h_prev and bb_pos < 0.5:
            return -float(np.clip(1.0 - bb_pos, 0.0, 1.0))
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "macd", "params": {"fast": self.macd_fast, "slow": self.macd_slow, "signal": 9}},
            {"name": "bollinger_bands", "params": {"period": self.bb_period, "num_std": 2.0}},
        ]
