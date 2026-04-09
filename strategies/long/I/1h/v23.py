"""MildTrendLongI1hV23 — Coppock Curve + OI Flow.

Economic logic: The Coppock Curve is a weighted moving average of two
rate-of-change values, originally designed to identify long-term buying
opportunities when the curve turns up from below zero. In 1H iron ore,
it captures medium-term momentum shifts. OI Flow tracks directional open
interest changes weighted by price direction — a rising flow line indicates
positions being built in the direction of price. When Coppock turns bullish
and OI flow confirms institutional accumulation, the trend has fundamental
backing from futures positioning.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.coppock import coppock
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV23(TrendingStrategy):
    """Coppock bullish turn + OI flow accumulation confirmation.

    Signal logic:
        Coppock > 0 AND rising AND oi_flow > oi_signal -> strong (0.7-1.0)
        Coppock > 0 or turning up                      -> weak (0.3-0.5)
        else                                           -> 0.0
    """

    name = "long_I_1h_v23"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 40  # max(roc_long=14) + wma(10) + buffer

    # Optimizable parameters
    wma_period: int = 10
    roc_long: int = 14
    oi_flow_period: int = 20
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
        self._coppock = coppock(
            self._closes, wma_period=self.wma_period, roc_long=self.roc_long,
        )
        self._oi_flow, self._oi_signal = oi_flow(
            self._closes, self._oi, self._volumes, period=self.oi_flow_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        cop = self._coppock[bar_index]
        flow = self._oi_flow[bar_index]
        flow_sig = self._oi_signal[bar_index]

        if np.isnan(cop):
            return 0.0

        # Check if Coppock is rising
        cop_prev = self._coppock[bar_index - 1] if bar_index > 0 else np.nan
        coppock_rising = not np.isnan(cop_prev) and cop > cop_prev

        # Coppock bullish: positive or turning up from below zero
        if cop <= 0.0 and not coppock_rising:
            return 0.0

        # Base signal from Coppock value and direction
        if cop > 0.0 and coppock_rising:
            base_signal = min(0.5, 0.3 + cop / 100.0)
        elif cop > 0.0:
            base_signal = 0.3
        else:
            # Turning up from below zero — early entry
            base_signal = 0.25

        # OI flow confirmation
        if not np.isnan(flow) and not np.isnan(flow_sig) and flow > flow_sig:
            flow_diff = flow - flow_sig
            oi_boost = min(0.4, 0.3 + abs(flow_diff) / (abs(flow_sig) + 1e-10) * 0.1)
            return min(1.0, base_signal + oi_boost)

        return min(0.5, base_signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "Coppock Curve", "array": self._coppock},
            {"name": "OI Flow", "array": self._oi_flow, "panel": "OI Flow"},
            {"name": "OI Signal", "array": self._oi_signal, "panel": "OI Flow"},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"OI Flow({self.oi_flow_period})",
                [
                    self._make_subplot_trace("OI Flow", datetimes, self._oi_flow, color="#bb86fc"),
                    self._make_subplot_trace("OI Signal", datetimes, self._oi_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            ),
            self._make_subplot(
                f"Coppock({self.wma_period},{self.roc_long})",
                [self._make_subplot_trace("Coppock", datetimes, self._coppock, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}

