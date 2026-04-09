"""MildTrendLongI1hV28 — TRIX + OI Momentum.

Economic logic: TRIX applies triple exponential smoothing then computes
the rate of change, filtering out short-term noise to reveal the underlying
trend direction. A positive TRIX above its signal line indicates sustained
bullish momentum. OI Momentum measures the rate of change of open interest
over a lookback period — rising OI during an uptrend means new positions are
being established in the trend direction (bullish continuation). When TRIX
confirms the trend and OI is growing, the move has staying power because
traders are adding new exposure rather than just covering old positions.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.trix import trix
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV28(TrendingStrategy):
    """TRIX bullish + OI momentum growth confirmation.

    Signal logic:
        TRIX > signal AND OI_mom > 0 -> strong (0.7-1.0)
        TRIX > signal only           -> moderate (0.3-0.5)
        TRIX > 0 only                -> weak (0.2-0.3)
        else                         -> 0.0
    """

    name = "long_I_1h_v28"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70  # 3*trix_period + signal(9) + buffer

    # Optimizable parameters
    trix_period: int = 15
    oi_mom_period: int = 20
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
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._trix_line, self._trix_signal = trix(
            self._closes, period=self.trix_period,
        )
        self._oi_mom = oi_momentum(self._oi, period=self.oi_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        trix_val = self._trix_line[bar_index]
        trix_sig = self._trix_signal[bar_index]
        oi_m = self._oi_mom[bar_index]

        if np.isnan(trix_val):
            return 0.0

        # TRIX must be positive for any long signal
        if trix_val <= 0.0:
            return 0.0

        # Base signal: TRIX above zero
        base_signal = min(0.3, 0.2 + abs(trix_val) * 0.01)

        # TRIX above signal line = stronger
        if not np.isnan(trix_sig) and trix_val > trix_sig:
            trix_spread = trix_val - trix_sig
            base_signal = min(0.5, 0.3 + trix_spread * 0.05)

            # OI momentum confirmation
            if not np.isnan(oi_m) and oi_m > 0.0:
                oi_boost = min(0.4, 0.2 + oi_m * 2.0)
                return min(1.0, base_signal + oi_boost)

            return base_signal

        return base_signal

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "TRIX Line", "array": self._trix_line, "panel": "TRIX"},
            {"name": "TRIX Signal", "array": self._trix_signal, "panel": "TRIX"},
            {"name": "OI Momentum", "array": self._oi_mom},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"TRIX({self.trix_period})",
                [
                    self._make_subplot_trace("TRIX", datetimes, self._trix_line, color="#bb86fc"),
                    self._make_subplot_trace("Signal", datetimes, self._trix_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            ),
            self._make_subplot(
                f"OI Mom({self.oi_mom_period})",
                [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
