"""Mean Reversion V1 — Bollinger Band + RSI Extreme.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: When price reaches Bollinger Band extremes AND RSI confirms
overbought/oversold, the probability of a short-term reversal increases.
This captures the over-extension phase where momentum traders push price
too far from fair value, creating a snapping-back opportunity.

Who loses money: Late momentum chasers who buy (sell) at extremes and
are forced to exit when price reverts to the mean.
"""

from __future__ import annotations

import numpy as np

from strategies.templates.mean_reversion_template import MeanReversionStrategy
from indicators.volatility.bollinger import bollinger_bands
from indicators.momentum.rsi import rsi


class MeanReversionV1(MeanReversionStrategy):
    """Bollinger Band + RSI extreme mean-reversion strategy.

    Full signal when price is outside Bollinger Bands AND RSI is at
    an extreme.  Partial signal when only one condition is met.

    Attributes:
        bb_period:       Bollinger Band SMA period.
        bb_std:          Bollinger Band standard deviation multiplier.
        rsi_period:      RSI lookback period.
        rsi_oversold:    RSI oversold threshold.
        rsi_overbought:  RSI overbought threshold.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "mr_v1"
    regime = "mean_reversion"
    horizon = None
    signal_dimensions = ["technical"]
    warmup: int = 30

    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    chandelier_mult: float = 2.0

    # --- Precomputed arrays ---
    _bb_upper: np.ndarray | None = None
    _bb_middle: np.ndarray | None = None
    _bb_lower: np.ndarray | None = None
    _rsi: np.ndarray | None = None

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
        """Precompute Bollinger Bands and RSI arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._bb_upper, self._bb_middle, self._bb_lower = bollinger_bands(
            self._closes, period=self.bb_period, num_std=self.bb_std,
        )
        self._rsi = rsi(self._closes, period=self.rsi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Combine Bollinger Band and RSI extremes for reversal signal."""
        close = self._closes[bar_index]
        upper = self._bb_upper[bar_index]
        lower = self._bb_lower[bar_index]
        middle = self._bb_middle[bar_index]
        rsi_val = self._rsi[bar_index]

        if np.isnan(upper) or np.isnan(rsi_val):
            return 0.0

        bb_width = upper - lower
        if bb_width <= 0:
            return 0.0

        # Determine Bollinger Band condition
        bb_signal = 0.0
        if close <= lower:
            # Price below lower band → buy signal (revert up)
            bb_signal = min((lower - close) / bb_width + 0.5, 1.0)
        elif close >= upper:
            # Price above upper band → sell signal (revert down)
            bb_signal = max(-((close - upper) / bb_width + 0.5), -1.0)

        # Determine RSI condition
        rsi_signal = 0.0
        if rsi_val <= self.rsi_oversold:
            rsi_signal = (self.rsi_oversold - rsi_val) / self.rsi_oversold
        elif rsi_val >= self.rsi_overbought:
            rsi_signal = -(rsi_val - self.rsi_overbought) / (100.0 - self.rsi_overbought)

        # Both conditions → full signal
        if bb_signal != 0.0 and rsi_signal != 0.0:
            # Signals should agree in direction for full conviction
            if (bb_signal > 0 and rsi_signal > 0) or (bb_signal < 0 and rsi_signal < 0):
                return float(np.clip(bb_signal, -1.0, 1.0))

        # Single condition → half signal
        if bb_signal != 0.0:
            return float(np.clip(bb_signal * 0.5, -1.0, 1.0))
        if rsi_signal != 0.0:
            return float(np.clip(rsi_signal * 0.5, -1.0, 1.0))

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "bollinger_bands",
                "params": {"period": self.bb_period, "num_std": self.bb_std},
            },
            {"name": "rsi", "params": {"period": self.rsi_period}},
        ]
