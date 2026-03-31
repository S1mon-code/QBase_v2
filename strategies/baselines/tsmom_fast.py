"""TSMOM Fast Baseline — 1-month Time Series Momentum.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: Momentum persists over short horizons because of initial
under-reaction followed by herding behaviour.  The simplest expression is
the sign of the past N-bar cumulative return.

This is a *baseline* — every fast-horizon trending strategy must beat it
to justify its complexity.
"""

from __future__ import annotations

import numpy as np

from strategies.templates.trending_template import TrendingStrategy


class TSMOMFast(TrendingStrategy):
    """1-month TSMOM Baseline (Fast Horizon).

    Signal: past 20 bars cumulative return > 0  ->  +1, else -1.

    Attributes:
        lookback:        Number of bars for return calculation.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "tsmom_fast"
    horizon = "fast"
    signal_dimensions = ["momentum"]
    warmup: int = 25

    lookback: int = 20
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
