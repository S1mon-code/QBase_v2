"""MildTrendLongIDailyV15 — Aroon Oscillator + Force Index Trend Confirmation.

Economic logic: The Aroon oscillator measures how recently price made a new high
versus a new low within the lookback window, capturing medium-term cyclical
positioning. Force Index (Elder) combines price change, direction, and volume
into a single momentum-volume composite. When both indicators agree, the trend
is confirmed from both a price structure and a volume-force perspective.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV15(TrendingStrategy):
    """Aroon oscillator signal scaled by Force Index directional agreement.

    Signal logic:
        - Aroon oscillator scaled to [-1, 1] by dividing by 100
        - Force Index sign confirms direction:
            - Same direction: return aroon_signal (full)
            - Opposite direction: return 0.5 * aroon_signal (weak)

    Attributes:
        aroon_period:    Aroon indicator lookback period.
        fi_period:       Force Index EMA smoothing period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v15"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "technical"]
    warmup: int = 58  # aroon_period(25) + fi_period(13) + 20

    aroon_period: int = 25
    fi_period: int = 13
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
        """Precompute Aroon oscillator and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        _, _, self._aroon_osc = aroon(
            self._highs,
            self._lows,
            period=self.aroon_period,
        )
        self._force_index = force_index(
            self._closes,
            self._volumes,
            period=self.fi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return Aroon-based signal confirmed by Force Index direction."""
        aroon_osc = self._aroon_osc[bar_index]
        fi = self._force_index[bar_index]

        if np.isnan(aroon_osc) or np.isnan(fi):
            return 0.0

        # Scale Aroon oscillator from [-100, 100] to [-1, 1]
        aroon_signal = aroon_osc / 100.0

        # Force Index directional sign
        fi_sign = 1.0 if fi > 0.0 else -1.0 if fi < 0.0 else 0.0

        if fi_sign == 0.0:
            return 0.5 * aroon_signal

        # Full signal if both indicators agree in direction; half signal otherwise
        if aroon_signal * fi_sign > 0:
            return aroon_signal
        return 0.5 * aroon_signal

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "aroon", "params": {"period": self.aroon_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"Aroon Osc({self.aroon_period})",
                [self._make_subplot_trace("Aroon Osc", datetimes, self._aroon_osc, color="#bb86fc")],
                zero_line=True,
            ),
            self._make_subplot(
                f"Force Index({self.fi_period})",
                [self._make_subplot_trace("Force Index", datetimes, self._force_index, color="#4fc3f7")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
