"""StrongTrendShortAGDailyV20 — TRIX(18) bearish + Stochastic(18,5) < 30 + CMF(20) < 0.

Economic logic: TRIX(18) below its signal line filters noise via triple
exponential smoothing, confirming genuine daily momentum shift. Stochastic
below 30 confirms oversold-trending conditions. CMF(20) negative validates
distribution. The triple confirmation reduces false signals in volatile silver.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.stochastic import stochastic
from indicators.momentum.trix import trix
from indicators.trend.ema import ema
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV20(TrendingStrategy):
    """TRIX(18) bearish + Stochastic(18,5) < 30 + CMF(20) < 0."""

    name = "strong_trend_short_AG_daily_v20"
    horizon = "slow"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    trix_period: int = 18
    stoch_k: int = 18
    stoch_d: int = 5
    cmf_period: int = 20
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes):
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._trix_line, self._trix_signal = trix(self._closes, self.trix_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes, k_period=self.stoch_k, d_period=self.stoch_d,
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes, self.cmf_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=5)

    def _raw_signal(self, i: int) -> float:
        tl = self._trix_line[i]
        ts = self._trix_signal[i]
        sk = self._stoch_k[i]
        cf = self._cmf[i]

        if any(np.isnan(v) for v in (tl, ts, sk, cf)):
            return 0.0

        if tl >= ts or sk >= 30.0 or cf >= 0:
            return 0.0

        strength = -0.45
        if sk < 20.0:
            strength -= 0.20
        if cf < -0.10:
            strength -= 0.20
        return max(-1.0, strength)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self):
        return [
            {"name": "TRIX", "array": self._trix_line, "panel": "TRIX"},
            {"name": "TRIX Signal", "array": self._trix_signal, "panel": "TRIX", "color": "#ff9800"},
            {"name": f"Stoch K({self.stoch_k})", "array": self._stoch_k,
             "panel": "Stoch", "y_range": [0, 100], "horizontal_lines": [20, 30, 80]},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "panel": "CMF", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(f"TRIX({self.trix_period})", [
                    self._make_subplot_trace("TRIX", datetimes, self._trix_line, color="#42a5f5"),
                    self._make_subplot_trace("Signal", datetimes, self._trix_signal, color="#ff9800"),
                ], zero_line=True),
                self._make_subplot(f"Stochastic({self.stoch_k},{self.stoch_d})", [
                    self._make_subplot_trace("K", datetimes, self._stoch_k, color="#bb86fc"),
                    self._make_subplot_trace("D", datetimes, self._stoch_d, color="#ffab40"),
                ], horizontal_lines=[20, 30, 80], y_range=[0, 100]),
                self._make_subplot(f"CMF({self.cmf_period})", [
                    self._make_subplot_trace("CMF", datetimes, self._cmf, color="#7e57c2"),
                ], zero_line=True),
            ],
        }
