"""MildTrendShortIDailyV1 — ADX Directional + OBV Decline.

Economic logic: When ADX confirms directional movement with DI- dominant
over DI+, and On-Balance Volume is declining below its SMA, the confluence
of bearish momentum and distribution pressure supports a short bias.
Mild trend = weaker signals with slope-based confirmation.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx_with_di
from indicators.volume.obv import obv
from indicators.trend.sma import sma
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV1(TrendingStrategy):
    """ADX(25) > 18 + DI-(25) > DI+ + OBV declining (OBV < SMA(40))."""

    name = "mild_trend_short_I_daily_v1"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    # Optimizable parameters (<=5 including chandelier_mult)
    adx_period: int = 25
    obv_sma_period: int = 40
    adx_threshold: float = 18.0
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
        self._adx_vals, self._plus_di, self._minus_di = adx_with_di(
            self._highs, self._lows, self._closes, self.adx_period,
        )
        self._obv = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv, self.obv_sma_period)

        # Pre-smooth raw signal
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        adx_val = self._adx_vals[bar_index]
        plus_di = self._plus_di[bar_index]
        minus_di = self._minus_di[bar_index]
        obv_val = self._obv[bar_index]
        obv_sma_val = self._obv_sma[bar_index]

        if any(np.isnan(v) for v in (adx_val, plus_di, minus_di, obv_sma_val)):
            return 0.0

        if adx_val <= self.adx_threshold:
            return 0.0
        if minus_di <= plus_di:
            return 0.0

        # Base signal from DI spread
        di_spread = (minus_di - plus_di) / max(minus_di + plus_di, 1.0)
        base = -0.3 - di_spread * 0.4  # mild: -0.3 to -0.7

        # OBV confirmation boost
        if obv_val < obv_sma_val:
            base -= 0.1

        return np.clip(base, -0.7, 0.0)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"ADX({self.adx_period})", "array": self._adx_vals,
             "type": "subplot", "y_range": [0, 100],
             "horizontal_lines": [self.adx_threshold]},
            {"name": f"+DI({self.adx_period})", "array": self._plus_di,
             "type": "subplot", "panel": f"ADX({self.adx_period})",
             "color": "#4caf50"},
            {"name": f"-DI({self.adx_period})", "array": self._minus_di,
             "type": "subplot", "panel": f"ADX({self.adx_period})",
             "color": "#f44336"},
            {"name": "OBV", "array": self._obv, "type": "subplot", "color": "#2196f3"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma,
             "type": "subplot", "panel": "OBV", "color": "#ff9800"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    f"ADX({self.adx_period})",
                    [
                        self._make_subplot_trace("ADX", datetimes, self._adx_vals, color="#bb86fc"),
                        self._make_subplot_trace("+DI", datetimes, self._plus_di, color="#4caf50"),
                        self._make_subplot_trace("-DI", datetimes, self._minus_di, color="#f44336"),
                    ],
                    horizontal_lines=[self.adx_threshold], y_range=[0, 100],
                ),
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv, color="#2196f3"),
                        self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ff9800"),
                    ],
                ),
            ],
        }
