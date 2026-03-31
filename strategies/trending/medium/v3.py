"""Trend Medium V3 — MACD Line + OI Flow.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: The MACD line (fast EMA - slow EMA) identifies trend direction:
positive means short-term momentum outpaces long-term momentum (bullish).  OI
Flow reveals whether institutional players are accumulating positions in the
same direction.  When both agree, conviction is high.

Who loses money: Traders who follow price moves without position building
confirmation — retail breakout chasers without institutional backing.
"""

from __future__ import annotations

import numpy as np

from strategies.templates.trending_template import TrendingStrategy
from indicators.momentum.macd import macd
from indicators.volume.oi_flow import oi_flow


class TrendMediumV3(TrendingStrategy):
    """MACD line direction + OI Flow directional confirmation.

    MACD line (fast_ema - slow_ema) gives the trend direction: positive for
    bullish, negative for bearish.  OI Flow confirms whether institutional
    capital is aligned with that direction.  Full signal when both agree;
    half signal when they diverge (MACD direction as tiebreaker).

    Attributes:
        macd_fast:       MACD fast EMA period.
        macd_slow:       MACD slow EMA period.
        macd_signal:     MACD signal line EMA period.
        oi_period:       OI Flow EMA smoothing period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "trend_medium_v3"
    horizon = "medium"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 150

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    oi_period: int = 14
    chandelier_mult: float = 2.5

    # --- Precomputed arrays ---
    _macd_line: np.ndarray | None = None
    _oi_flow_line: np.ndarray | None = None

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
        """Precompute MACD line and OI Flow arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._macd_line, _signal_line, _histogram = macd(
            self._closes,
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
        )
        self._oi_flow_line, _flow_signal = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Combine MACD line direction with OI Flow direction."""
        macd_val = self._macd_line[bar_index]
        flow = self._oi_flow_line[bar_index]

        if np.isnan(macd_val) or np.isnan(flow):
            return 0.0

        # MACD line > 0 means fast EMA above slow EMA → bullish trend
        macd_dir = 1.0 if macd_val > 0 else -1.0
        flow_dir = 1.0 if flow > 0 else -1.0

        # Full signal when both agree
        if macd_dir == flow_dir:
            return macd_dir

        # Half signal when they diverge — MACD direction as tiebreaker
        return macd_dir * 0.5

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "macd",
                "params": {
                    "fast": self.macd_fast,
                    "slow": self.macd_slow,
                    "signal": self.macd_signal,
                },
            },
            {
                "name": "oi_flow",
                "params": {"period": self.oi_period},
            },
        ]
