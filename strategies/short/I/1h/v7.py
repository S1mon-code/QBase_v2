"""MildTrendShortI1hV7 — ADX trending + DI- dominant + Chaikin Osc negative.

Economic logic: ADX(14) > 18 confirms trend is present (mild threshold for mild
regime). DI- > DI+ validates bearish directional energy. Chaikin Oscillator(8,16)
below zero confirms distribution momentum in the A/D line. Signal smoothed with
EMA(3) to reduce noise in mild-trend conditions.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx_with_di
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV7(TrendingStrategy):
    """ADX(14) > 18 + DI- > DI+ + Chaikin Osc(8,16) < 0."""

    name = "short_I_1h_v7"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 35

    adx_period: int = 14
    chaikin_fast: int = 8
    chaikin_slow: int = 16
    chandelier_mult: float = 2.8

    _adx_arr: np.ndarray | None = None
    _plus_di: np.ndarray | None = None
    _minus_di: np.ndarray | None = None
    _chaikin_arr: np.ndarray | None = None
    _smooth_signal: np.ndarray | None = None

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
        self._adx_arr, self._plus_di, self._minus_di = adx_with_di(
            self._highs, self._lows, self._closes, self.adx_period,
        )
        self._chaikin_arr = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            self.chaikin_fast, self.chaikin_slow,
        )

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        a = self._adx_arr[bar_index]
        dp = self._plus_di[bar_index]
        dm = self._minus_di[bar_index]
        ch = self._chaikin_arr[bar_index]

        if any(np.isnan(v) for v in (a, dp, dm, ch)):
            return 0.0

        if a < 18 or dm <= dp:
            return 0.0

        di_spread = (dm - dp) / (dm + dp) if (dm + dp) > 0 else 0.0
        signal = -(0.3 + min(0.2, di_spread * 1.5))

        if ch < 0:
            signal -= 0.15

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "ADX", "array": self._adx_arr, "type": "subplot", "panel": "ADX",
             "y_range": [0, 100], "horizontal_lines": [18]},
            {"name": "+DI", "array": self._plus_di, "type": "subplot", "panel": "ADX", "color": "#66bb6a"},
            {"name": "-DI", "array": self._minus_di, "type": "subplot", "panel": "ADX", "color": "#ef5350"},
            {"name": f"Chaikin({self.chaikin_fast},{self.chaikin_slow})", "array": self._chaikin_arr,
             "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    "ADX / DI",
                    [
                        self._make_subplot_trace("ADX", datetimes, self._adx_arr, color="#ffab40"),
                        self._make_subplot_trace("+DI", datetimes, self._plus_di, color="#66bb6a"),
                        self._make_subplot_trace("-DI", datetimes, self._minus_di, color="#ef5350"),
                    ],
                    horizontal_lines=[18], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"Chaikin({self.chaikin_fast},{self.chaikin_slow})",
                    [self._make_subplot_trace("Chaikin", datetimes, self._chaikin_arr, color="#42a5f5")],
                    zero_line=True,
                ),
            ],
        }
