"""MildTrendLongI2hV16 — Bollinger Band Width Squeeze + MACD Direction.

Economic logic: Bollinger Band squeezes (historically low volatility) precede
explosive directional moves. By detecting when BB width is at its lowest
percentile and then using MACD to determine breakout direction, this strategy
captures the high-probability expansion phase. Outside squeezes, MACD still
provides a reduced-confidence directional signal.
"""

from __future__ import annotations

import numpy as np

from indicators.volatility.bollinger import bollinger_bands
from indicators.momentum.macd import macd
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI2hV16(TrendingStrategy):
    """BB width squeeze detection with MACD directional confirmation.

    Signal logic:
        - BB width at <15th percentile in last 5 bars (squeeze) + MACD > 0 → +1.0
        - Squeeze + MACD < 0 → -1.0
        - No squeeze → MACD direction * 0.3

    Attributes:
        bb_period:       Bollinger Bands period.
        macd_fast:       MACD fast EMA period.
        macd_slow:       MACD slow EMA period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mild_trend_long_I_2h_v16"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 146  # max(bb_period, macd_slow) + 100 + 20

    bb_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
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
        """Precompute BB, BB width percentile, and MACD arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            self._closes, period=self.bb_period,
        )
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            self._closes, fast=self.macd_fast, slow=self.macd_slow,
        )

        # Precompute BB width and rolling percentile rank over 100 bars
        n = len(self._closes)
        self._bb_width = np.full(n, np.nan)
        self._bb_width_pct = np.full(n, np.nan)

        for i in range(n):
            mid_val = self._bb_mid[i]
            if np.isnan(mid_val) or mid_val == 0.0:
                continue
            self._bb_width[i] = (self._bb_upper[i] - self._bb_lower[i]) / mid_val

        for i in range(100, n):
            window = self._bb_width[i - 100 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) < 10:
                continue
            current = self._bb_width[i]
            if np.isnan(current):
                continue
            self._bb_width_pct[i] = np.sum(valid < current) / len(valid)

    def _generate_signal(self, bar_index: int) -> float:
        """Return squeeze-aware MACD directional signal."""
        macd_val = self._macd_line[bar_index]
        pct_val = self._bb_width_pct[bar_index]

        if np.isnan(macd_val):
            return 0.0

        # Check for recent squeeze: any bar in last 5 with percentile < 0.15
        recent_squeeze = False
        start = max(0, bar_index - 4)
        for i in range(start, bar_index + 1):
            p = self._bb_width_pct[i]
            if not np.isnan(p) and p < 0.15:
                recent_squeeze = True
                break

        if recent_squeeze:
            if macd_val > 0.0:
                return 1.0
            if macd_val < 0.0:
                return -1.0
            return 0.0

        # No squeeze: weak MACD directional signal
        if macd_val > 0.0:
            return 0.3
        if macd_val < 0.0:
            return -0.3
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "bollinger_bands", "params": {"period": self.bb_period}},
            {
                "name": "macd",
                "params": {"fast": self.macd_fast, "slow": self.macd_slow},
            },
        ]
