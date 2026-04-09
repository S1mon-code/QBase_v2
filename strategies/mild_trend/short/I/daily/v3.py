"""MildTrendShortIDailyV3 — EMA Slope + Aroon Down + Force Index.

Economic logic: EMA(80) slope turning negative confirms a structural
downtrend. Aroon Down exceeding 60 validates that recent lows are being
made frequently. Negative Force Index adds volume-weighted selling
pressure confirmation for the mild short setup.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.aroon import aroon
from indicators.volume.force_index import force_index
from indicators.trend.linear_regression import linear_regression_slope
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV3(TrendingStrategy):
    """EMA(80) slope < 0 + Aroon Down(40) > 60 + Force Index(25) < 0."""

    name = "mild_trend_short_I_daily_v3"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 100

    # Optimizable parameters (<=5 including chandelier_mult)
    ema_period: int = 80
    slope_lookback: int = 10
    aroon_period: int = 40
    fi_period: int = 25
    chandelier_mult: float = 3.5

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
        self._ema_line = ema(self._closes, self.ema_period)
        self._ema_slope = linear_regression_slope(self._ema_line, self.slope_lookback)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            self._highs, self._lows, self.aroon_period,
        )
        self._fi = force_index(self._closes, self._volumes, self.fi_period)

        # Pre-smooth raw signal
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        slope = self._ema_slope[bar_index]
        aroon_d = self._aroon_down[bar_index]
        fi_val = self._fi[bar_index]

        if any(np.isnan(v) for v in (slope, aroon_d, fi_val)):
            return 0.0

        # EMA slope must be negative
        if slope >= 0:
            return 0.0

        # Aroon Down must show dominance
        if aroon_d <= 60:
            return 0.0

        # Slope magnitude drives base signal
        slope_strength = min(abs(slope) / 2.0, 1.0)
        base = -0.3 - slope_strength * 0.3  # -0.3 to -0.6

        # Force Index confirmation
        if fi_val < 0:
            base -= 0.1

        return np.clip(base, -0.7, 0.0)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"EMA({self.ema_period})", "array": self._ema_line,
             "type": "overlay", "color": "#ffab40"},
            {"name": "Aroon Down", "array": self._aroon_down,
             "type": "subplot", "panel": f"Aroon({self.aroon_period})",
             "y_range": [0, 100], "horizontal_lines": [60], "color": "#f44336"},
            {"name": "Aroon Up", "array": self._aroon_up,
             "type": "subplot", "panel": f"Aroon({self.aroon_period})",
             "color": "#4caf50"},
            {"name": f"Force Index({self.fi_period})", "array": self._fi,
             "type": "subplot", "zero_line": True, "color": "#2196f3"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_period})", datetimes, self._ema_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Aroon({self.aroon_period})",
                    [
                        self._make_subplot_trace("Aroon Up", datetimes, self._aroon_up, color="#4caf50"),
                        self._make_subplot_trace("Aroon Down", datetimes, self._aroon_down, color="#f44336"),
                    ],
                    horizontal_lines=[60], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"Force Index({self.fi_period})",
                    [self._make_subplot_trace("FI", datetimes, self._fi, color="#2196f3")],
                    zero_line=True,
                ),
            ],
        }
