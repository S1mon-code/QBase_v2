"""Trend Slow V1 — Donchian Channel Breakout.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: Donchian breakout is the classic trend-following entry
(Turtle Trading).  A close above the N-period highest high indicates a
new long-term trend.  The width of the channel relative to price provides
a secondary confidence measure — wider channels mean bigger breakouts.

Who loses money: Range-bound traders who sell resistance and buy support
get stopped out when the market breaks into a genuine new trend.
"""

from __future__ import annotations

import numpy as np

from strategies.templates.trending_template import TrendingStrategy
from indicators.trend.donchian import donchian


class TrendSlowV1(TrendingStrategy):
    """Donchian Channel breakout strategy for slow horizon.

    Signal is +1 when close breaks above upper channel, -1 when close
    breaks below lower channel.  Signal intensity scales with how far
    price has moved beyond the channel boundary.

    Attributes:
        dc_period:       Donchian Channel lookback period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "trend_slow_v1"
    horizon = "slow"
    signal_dimensions = ["momentum"]
    warmup: int = 260

    dc_period: int = 200
    chandelier_mult: float = 3.0

    # --- Precomputed arrays ---
    _dc_upper: np.ndarray | None = None
    _dc_lower: np.ndarray | None = None
    _dc_middle: np.ndarray | None = None

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
        """Precompute Donchian Channel arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._dc_upper, self._dc_lower, self._dc_middle = donchian(
            self._highs, self._lows, period=self.dc_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal based on Donchian Channel breakout."""
        upper = self._dc_upper[bar_index]
        lower = self._dc_lower[bar_index]
        middle = self._dc_middle[bar_index]
        close = self._closes[bar_index]

        if np.isnan(upper) or np.isnan(lower) or np.isnan(middle):
            return 0.0

        channel_width = upper - lower
        if channel_width <= 0:
            return 0.0

        # Breakout above upper channel
        if close >= upper:
            # Scale by how far above the midpoint
            excess = (close - middle) / channel_width
            return min(excess * 2.0, 1.0)

        # Breakout below lower channel
        if close <= lower:
            excess = (middle - close) / channel_width
            return max(-excess * 2.0, -1.0)

        # Inside channel — position relative to midpoint
        position = (close - middle) / (channel_width / 2.0)
        return float(np.clip(position * 0.3, -0.5, 0.5))

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "donchian", "params": {"period": self.dc_period}},
        ]
