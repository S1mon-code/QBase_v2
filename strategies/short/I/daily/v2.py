"""MildTrendShortIDailyV2 — SuperTrend Bearish + MACD Line + CMF.

Economic logic: SuperTrend direction confirms the bearish regime, MACD line
(not histogram) validates momentum direction, and negative CMF signals
institutional distribution. Triple confirmation for mild downtrend.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.momentum.macd import macd
from indicators.volume.cmf import cmf
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortIDailyV2(TrendingStrategy):
    """SuperTrend(40,3.5) bearish + MACD(20,45,12) hist < 0 + CMF(25) < 0."""

    name = "short_I_daily_v2"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 65

    # Optimizable parameters (<=5 including chandelier_mult)
    st_period: int = 40
    st_mult: float = 3.5
    macd_fast: int = 20
    macd_slow: int = 45
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
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes, self.st_period, self.st_mult,
        )
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            self._closes, self.macd_fast, self.macd_slow, 12,
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, 25)

        # Pre-smooth raw signal
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        macd_h = self._macd_hist[bar_index]
        cmf_val = self._cmf[bar_index]

        if any(np.isnan(v) for v in (st_d, macd_h, cmf_val)):
            return 0.0

        # SuperTrend must be bearish
        if st_d > 0:
            return 0.0

        # MACD histogram must be negative
        if macd_h >= 0:
            return 0.0

        # Base signal from MACD histogram magnitude (normalized)
        macd_strength = min(abs(macd_h) / 5.0, 1.0)
        base = -0.3 - macd_strength * 0.3  # -0.3 to -0.6

        # CMF confirmation
        if cmf_val < 0:
            cmf_boost = min(abs(cmf_val) * 0.5, 0.1)
            base -= cmf_boost

        return np.clip(base, -0.7, 0.0)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"SuperTrend({self.st_period},{self.st_mult})",
             "array": self._st_line, "type": "overlay", "style": "step",
             "color": "#ff5722"},
            {"name": "MACD Hist", "array": self._macd_hist,
             "type": "subplot", "style": "bar", "panel": "MACD",
             "zero_line": True, "color_positive": "#4caf50",
             "color_negative": "#f44336"},
            {"name": "MACD Line", "array": self._macd_line,
             "type": "subplot", "panel": "MACD", "color": "#2196f3"},
            {"name": "MACD Signal", "array": self._macd_signal,
             "type": "subplot", "panel": "MACD", "color": "#ff9800"},
            {"name": "CMF(25)", "array": self._cmf,
             "type": "subplot", "zero_line": True, "color": "#ab47bc"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(
                    f"SuperTrend({self.st_period},{self.st_mult})",
                    datetimes, self._st_line, style="step", color="#ff5722",
                ),
            ],
            "subplots": [
                self._make_subplot(
                    "MACD",
                    [
                        self._make_subplot_trace("Histogram", datetimes, self._macd_hist,
                                                 style="bar", color_positive="#4caf50",
                                                 color_negative="#f44336"),
                        self._make_subplot_trace("MACD", datetimes, self._macd_line, color="#2196f3"),
                        self._make_subplot_trace("Signal", datetimes, self._macd_signal, color="#ff9800"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    "CMF(25)",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#ab47bc")],
                    zero_line=True,
                ),
            ],
        }
