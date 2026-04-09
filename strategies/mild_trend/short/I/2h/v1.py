"""MildTrendShortI2hV1 — EMA slope + ADX + DI- dominance + CMF distribution.

Economic logic: EMA(25) declining slope identifies intermediate bearish drift.
ADX(18) > 18 confirms a directional move is underway (mild threshold for mild
trends). DI- > DI+ validates bears are in control. CMF(18) < 0 shows
institutional distribution — money flowing out of the instrument.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.adx import adx_with_di
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV1(TrendingStrategy):
    """EMA(25) slope < 0 + ADX(18) > 18 + DI- > DI+ + CMF(18) < 0."""

    name = "mild_trend_short_I_2h_v1"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 55

    ema_period: int = 25
    adx_period: int = 18
    cmf_period: int = 18
    chandelier_mult: float = 3.0

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
        self._adx_vals, self._di_plus, self._di_minus = adx_with_di(
            self._highs, self._lows, self._closes, self.adx_period,
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        # Signal smoothing
        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        from indicators.trend.ema import ema as _ema
        self._smooth_signal = _ema(raw, period=4)

    def _raw_signal(self, bar_index: int) -> float:
        i = bar_index
        e_now = self._ema_line[i]
        e_prev = self._ema_line[i - 1]
        adx_val = self._adx_vals[i]
        di_p = self._di_plus[i]
        di_m = self._di_minus[i]
        cmf_val = self._cmf[i]

        if any(np.isnan(v) for v in (e_now, e_prev, adx_val, di_p, di_m)):
            return 0.0

        ema_slope = (e_now - e_prev) / e_prev if e_prev != 0 else 0.0
        if ema_slope >= 0:
            return 0.0
        if adx_val <= 18:
            return 0.0
        if di_m <= di_p:
            return 0.0

        # Base signal from slope magnitude
        slope_str = min(1.0, abs(ema_slope) * 500)
        signal = -(0.30 + slope_str * 0.15)

        # DI spread boost
        di_spread = (di_m - di_p) / (di_m + di_p) if (di_m + di_p) > 0 else 0.0
        signal -= 0.10 * min(1.0, di_spread * 3.0)

        # CMF confirmation
        if not np.isnan(cmf_val) and cmf_val < 0:
            signal -= 0.10 * min(1.0, abs(cmf_val) * 5.0)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"EMA({self.ema_period})", "array": self._ema_line, "type": "overlay", "color": "#ffab40"},
            {"name": f"ADX({self.adx_period})", "array": self._adx_vals, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [18, 25]},
            {"name": f"+DI({self.adx_period})", "array": self._di_plus, "type": "subplot",
             "panel": f"DI({self.adx_period})", "color": "#66bb6a"},
            {"name": f"-DI({self.adx_period})", "array": self._di_minus, "type": "subplot",
             "panel": f"DI({self.adx_period})", "color": "#ef5350"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"EMA({self.ema_period})", datetimes, self._ema_line, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"ADX({self.adx_period})",
                    [self._make_subplot_trace("ADX", datetimes, self._adx_vals, color="#42a5f5")],
                    horizontal_lines=[18, 25], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"DI({self.adx_period})",
                    [
                        self._make_subplot_trace("+DI", datetimes, self._di_plus, color="#66bb6a"),
                        self._make_subplot_trace("-DI", datetimes, self._di_minus, color="#ef5350"),
                    ],
                ),
                self._make_subplot(
                    f"CMF({self.cmf_period})",
                    [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#ab47bc")],
                    zero_line=True,
                ),
            ],
        }
