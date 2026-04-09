"""MildTrendLongIDailyV35 — Aroon Trend Dominance + Force Index Sustained Momentum.

Economic logic: Aroon Up above 70 means price made a new high within the last
30% of the lookback window — a textbook sign of a strong uptrend. Force Index
combines price change direction and volume into a single momentum-flow measure;
a positive Force Index above zero confirms that bulls are pressing with volume
support. The combination of price-structure strength (Aroon) and volume-driven
momentum (Force Index) yields high-quality trend entries.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.aroon import aroon
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV35(TrendingStrategy):
    """Aroon dominance confirmed by Force Index momentum direction.

    Signal logic:
        - Aroon up > 70 AND Force Index > 0: +1.0
        - Aroon down > 70 AND Force Index < 0: -1.0
        - Aroon up > 50 AND Aroon down < 50 AND FI > 0: +0.7
        - Aroon down > 50 AND Aroon up < 50 AND FI < 0: -0.7
        - Otherwise: 0.0

    Attributes:
        aroon_period:    Aroon lookback period.
        fi_period:       Force Index EMA smoothing period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v35"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
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
        """Precompute Aroon and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._aroon_up, self._aroon_dn, _ = aroon(
            self._highs,
            self._lows,
            period=self.aroon_period,
        )
        self._fi = force_index(
            self._closes,
            self._volumes,
            period=self.fi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on Aroon dominance and Force Index sign."""
        aroon_up = self._aroon_up[bar_index]
        aroon_dn = self._aroon_dn[bar_index]
        fi = self._fi[bar_index]

        if np.isnan(aroon_up) or np.isnan(aroon_dn) or np.isnan(fi):
            return 0.0

        if aroon_up > 70 and fi > 0:
            return 1.0
        if aroon_dn > 70 and fi < 0:
            return -1.0
        if aroon_up > 50 and aroon_dn < 50 and fi > 0:
            return 0.7
        if aroon_dn > 50 and aroon_up < 50 and fi < 0:
            return -0.7
        return 0.0

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
                f"Aroon({self.aroon_period})",
                [
                    self._make_subplot_trace("Aroon Up", datetimes, self._aroon_up, color="#26a69a"),
                    self._make_subplot_trace("Aroon Down", datetimes, self._aroon_dn, color="#ef5350"),
                ],
                y_range=[0, 100],
            ),
            self._make_subplot(
                f"Force Index({self.fi_period})",
                [self._make_subplot_trace("Force Index", datetimes, self._fi, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}
