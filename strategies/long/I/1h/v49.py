"""MildTrendLongI1hV49 — CCI + Force Index Confirmation.

Economic logic: Commodity Channel Index measures how far price deviates from its
statistical mean — readings above +100 indicate strong bullish momentum, below
-100 strong bearish momentum. Force Index (price change x volume) confirms that
the move carries genuine volume participation. This two-layer filter is
particularly effective for iron ore, where CCI extremes often precede sustained
directional moves when backed by volume.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV49(TrendingStrategy):
    """CCI threshold levels confirmed by Force Index direction.

    Signal logic:
        - CCI > 100 AND FI > 0:  +1.0
        - CCI < -100 AND FI < 0: -1.0
        - CCI in (0, 100) AND FI > 0:  +0.5
        - CCI in (-100, 0) AND FI < 0: -0.5
        - Else: 0.0
    """

    name = "long_I_1h_v49"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 47  # cci_period(14) + fi_period(13) + 20

    cci_period: int = 14
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
        """Precompute CCI and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._cci = cci(
            self._highs, self._lows, self._closes, period=self.cci_period,
        )
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on CCI levels and Force Index."""
        cci_val = self._cci[bar_index]
        fi_val = self._fi[bar_index]

        if np.isnan(cci_val) or np.isnan(fi_val):
            return 0.0

        if cci_val > 100 and fi_val > 0:
            return 1.0
        if cci_val < -100 and fi_val < 0:
            return -1.0
        if 0 < cci_val <= 100 and fi_val > 0:
            return 0.5
        if -100 <= cci_val < 0 and fi_val < 0:
            return -0.5
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "cci", "params": {"period": self.cci_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"CCI({self.cci_period})",
                [self._make_subplot_trace("CCI", datetimes, self._cci, color="#bb86fc")],
                horizontal_lines=[-100, 100],
            ),
            self._make_subplot(
                f"Force Index({self.fi_period})",
                [self._make_subplot_trace("Force Index", datetimes, self._fi, color="#4fc3f7")],
                zero_line=True,
            )
        ]
        return {"overlays": [], "subplots": subplots}

