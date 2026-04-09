"""MildTrendLongI1hV12 — SuperTrend + Force Index.

Economic logic: SuperTrend captures the dominant trend regime via ATR-based
trailing bands, while Force Index (price change * volume) confirms that
institutional money is flowing in the trend direction. The combination filters
out low-conviction moves where price drifts without volume commitment. Normalized
force index scaling ensures signal strength adapts to changing volatility regimes.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV12(TrendingStrategy):
    """SuperTrend uptrend + positive Force Index momentum confirmation.

    Signal logic:
        supertrend direction == 1 (up) AND force_index > 0
            -> signal = min(1.0, 0.5 + fi / rolling_max_abs_fi * 0.5)
        else -> 0.0
    """

    name = "long_I_1h_v12"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60  # st_period(30) + normalization window buffer

    # Optimizable parameters
    st_period: int = 30
    st_mult: float = 2.0
    fi_period: int = 20
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
        """Precompute SuperTrend and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._st_values, self._st_direction = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

        # Precompute rolling max of abs(force_index) over 50 bars for normalization
        n = len(self._fi)
        self._fi_max = np.full(n, np.nan)
        abs_fi = np.abs(self._fi)
        for i in range(49, n):
            window = abs_fi[i - 49 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._fi_max[i] = np.max(valid)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal based on SuperTrend direction with Force Index strength."""
        st_dir = self._st_direction[bar_index]
        fi_val = self._fi[bar_index]
        fi_max = self._fi_max[bar_index]

        if np.isnan(st_dir) or np.isnan(fi_val) or np.isnan(fi_max) or fi_max == 0.0:
            return 0.0

        if st_dir == 1.0 and fi_val > 0.0:
            normalized_fi = fi_val / fi_max
            return min(1.0, 0.5 + normalized_fi * 0.5)

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for auto-generated panels."""
        return [
            {"name": "SuperTrend", "array": self._st_values, "style": "step"},
            {"name": "Force Index", "array": self._fi},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"SuperTrend({self.st_period})", datetimes, self._st_values, style="step", color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"Force Index({self.fi_period})",
                [self._make_subplot_trace("Force Index", datetimes, self._fi, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}
