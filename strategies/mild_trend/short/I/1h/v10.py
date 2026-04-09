"""MildTrendShortI1hV10 — Vortex bearish + MACD hist negative + OBV below EMA.

Economic logic: Vortex(14) VI- > VI+ on 1H confirms bearish rotational energy.
MACD histogram < 0 validates momentum is weakening or negative. OBV below its
EMA(16) shows volume-weighted distribution. Signal smoothed with EMA(3) to
prevent overtrading in mild downtrend conditions.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.vortex import vortex
from indicators.momentum.macd import macd
from indicators.volume.obv import obv
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV10(TrendingStrategy):
    """Vortex(14) VI- > VI+ + MACD hist < 0 + OBV < OBV_EMA(16)."""

    name = "mild_trend_short_I_1h_v10"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 35

    vortex_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    obv_ema_period: int = 16
    chandelier_mult: float = 3.2

    _vi_plus: np.ndarray | None = None
    _vi_minus: np.ndarray | None = None
    _macd_line: np.ndarray | None = None
    _macd_hist: np.ndarray | None = None
    _obv_arr: np.ndarray | None = None
    _obv_ema: np.ndarray | None = None
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
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, self.vortex_period,
        )
        self._macd_line, _, self._macd_hist = macd(
            self._closes, self.macd_fast, self.macd_slow, 9,
        )
        self._obv_arr = obv(self._closes, self._volumes)
        self._obv_ema = ema(self._obv_arr, self.obv_ema_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        vp = self._vi_plus[bar_index]
        vm = self._vi_minus[bar_index]
        mh = self._macd_hist[bar_index]
        ov = self._obv_arr[bar_index]
        ov_e = self._obv_ema[bar_index]

        if any(np.isnan(v) for v in (vp, vm, mh, ov, ov_e)):
            return 0.0

        if vm <= vp:
            return 0.0

        vi_spread = vm - vp
        signal = -(0.3 + min(0.15, vi_spread * 2.0))

        if mh < 0:
            signal -= 0.1

        if ov < ov_e:
            signal -= 0.1

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "VI+", "array": self._vi_plus, "type": "subplot", "panel": "Vortex",
             "color": "#66bb6a", "horizontal_lines": [1.0]},
            {"name": "VI-", "array": self._vi_minus, "type": "subplot", "panel": "Vortex", "color": "#ef5350"},
            {"name": "MACD Hist", "array": self._macd_hist, "type": "subplot", "panel": "MACD",
             "style": "bar", "color_positive": "#26a69a", "color_negative": "#ef5350", "zero_line": True},
            {"name": "OBV", "array": self._obv_arr, "type": "subplot", "panel": "OBV"},
            {"name": f"OBV EMA({self.obv_ema_period})", "array": self._obv_ema,
             "type": "subplot", "panel": "OBV", "color": "#ffab40"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    "Vortex",
                    [
                        self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#66bb6a"),
                        self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
                    ],
                    horizontal_lines=[1.0],
                ),
                self._make_subplot(
                    "MACD Hist",
                    [self._make_subplot_trace("Hist", datetimes, self._macd_hist, style="bar",
                                              color_positive="#26a69a", color_negative="#ef5350")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv_arr, color="#42a5f5"),
                        self._make_subplot_trace(f"OBV EMA({self.obv_ema_period})", datetimes,
                                                 self._obv_ema, color="#ffab40"),
                    ],
                ),
            ],
        }
