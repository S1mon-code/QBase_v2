"""Trend Fast V1 — Short Momentum + Volume Spike.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: Short-term price momentum (5-bar ROC) captures the initial
impulse of a move.  Volume spikes confirm that the move has institutional
participation rather than being random noise.  Combining the two filters
out low-conviction momentum signals that frequently reverse.

Who loses money: Market makers and mean-reversion traders caught off-guard
by sudden institutional-driven moves.
"""

from __future__ import annotations

import numpy as np

from strategies.templates.trending_template import TrendingStrategy
from indicators.momentum.roc import rate_of_change
from indicators.volume.volume_spike import volume_spike


class TrendFastV1(TrendingStrategy):
    """Short-term ROC momentum gated by volume spikes.

    Full signal when ROC exceeds threshold and volume spike is detected.
    Reduced signal on momentum alone without volume confirmation.

    Attributes:
        roc_period:        Rate-of-change lookback.
        roc_threshold:     Minimum absolute ROC for signal generation.
        vol_spike_period:  Volume spike rolling average period.
        vol_spike_mult:    Volume spike threshold multiplier.
        chandelier_mult:   Chandelier Exit multiplier (optimisable).
    """

    name = "trend_fast_v1"
    horizon = "fast"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 30

    roc_period: int = 5
    roc_threshold: float = 0.5
    vol_spike_period: int = 20
    vol_spike_mult: float = 2.0
    chandelier_mult: float = 2.5

    # --- Precomputed arrays ---
    _roc: np.ndarray | None = None
    _vol_spike: np.ndarray | None = None

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
        """Precompute ROC and volume spike arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._roc = rate_of_change(self._closes, period=self.roc_period)
        self._vol_spike = volume_spike(
            self._volumes,
            period=self.vol_spike_period,
            threshold=self.vol_spike_mult,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Combine short-term momentum with volume spike confirmation."""
        roc_val = self._roc[bar_index]

        if np.isnan(roc_val):
            return 0.0

        # Determine direction
        if abs(roc_val) < self.roc_threshold:
            return 0.0

        direction = 1.0 if roc_val > 0 else -1.0

        # Scale by ROC magnitude (cap at 5% for max signal)
        magnitude = min(abs(roc_val) / 5.0, 1.0)

        # Boost on volume spike
        has_spike = bool(self._vol_spike[bar_index])
        if has_spike:
            return direction * magnitude

        # Reduced signal without volume confirmation
        return direction * magnitude * 0.5

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "rate_of_change", "params": {"period": self.roc_period}},
            {
                "name": "volume_spike",
                "params": {
                    "period": self.vol_spike_period,
                    "threshold": self.vol_spike_mult,
                },
            },
        ]
