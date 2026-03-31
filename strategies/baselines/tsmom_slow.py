"""TSMOM Slow Baseline — 12-month Time Series Momentum.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: Long-term momentum captures the full trend lifecycle — from
initial under-reaction through herding to eventual over-extension.  12-month
lookback is the upper bound of academic TSMOM signals.
"""

from __future__ import annotations

import numpy as np

from strategies.templates.trending_template import TrendingStrategy


class TSMOMSlow(TrendingStrategy):
    """12-month TSMOM Baseline (Slow Horizon).

    Signal: past 250 bars cumulative return > 0  ->  +1, else -1.

    Attributes:
        lookback:        Number of bars for return calculation.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "tsmom_slow"
    horizon = "slow"
    signal_dimensions = ["momentum"]
    warmup: int = 255

    lookback: int = 250
    chandelier_mult: float = 3.0

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
        """Precompute cumulative returns array."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        n = len(self._closes)
        self._cum_return = np.full(n, np.nan, dtype=np.float64)
        for i in range(self.lookback, n):
            prev = self._closes[i - self.lookback]
            if prev > 0:
                self._cum_return[i] = self._closes[i] / prev - 1.0

    def _generate_signal(self, bar_index: int) -> float:
        """Return +1 if cumulative return is positive, -1 otherwise."""
        cr = self._cum_return[bar_index]
        if np.isnan(cr):
            return 0.0
        return 1.0 if cr > 0 else -1.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "tsmom_return", "params": {"lookback": self.lookback}},
        ]
