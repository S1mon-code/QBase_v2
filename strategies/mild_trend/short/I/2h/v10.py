"""MildTrendShortI2hV10 — Keltner Lower Break + ROC Negative + Chaikin Osc.

Economic logic: Breaking below Keltner lower band on 2H signals volatility-adjusted
weakness. Negative ROC confirms declining prices. Chaikin Oscillator below zero
validates distribution momentum in the A/D line.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.keltner import keltner
from indicators.momentum.roc import rate_of_change
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV10(TrendingStrategy):
    """Break Keltner(50,2.0) lower + ROC(30)<0 + ChaikinOsc(15,40)<0."""

    name = "mild_trend_short_I_2h_v10"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    kelt_ema: int = 50
    kelt_mult: float = 2.0
    roc_period: int = 30
    co_fast: int = 15
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
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_ema, multiplier=self.kelt_mult,
        )
        self._roc = rate_of_change(self._closes, self.roc_period)
        self._chaikin = chaikin_oscillator(
            self._highs, self._lows, self._closes, self._volumes,
            self.co_fast, 40,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        kl = self._kelt_lower[bar_index]
        km = self._kelt_mid[bar_index]
        roc_val = self._roc[bar_index]
        co = self._chaikin[bar_index]

        if any(np.isnan(v) for v in (close, kl, roc_val)):
            return 0.0

        signal = 0.0

        if close < kl:
            signal = -0.4
        elif not np.isnan(km) and close < km:
            signal = -0.2
        else:
            return 0.0

        if roc_val < 0:
            roc_str = min(1.0, abs(roc_val) / 8.0)
            signal -= 0.2 * roc_str

        if not np.isnan(co) and co < 0:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"Kelt Upper({self.kelt_ema})", "array": self._kelt_upper, "type": "overlay", "style": "dash", "color": "#ef5350"},
            {"name": f"Kelt Mid({self.kelt_ema})", "array": self._kelt_mid, "type": "overlay", "color": "#ffab40"},
            {"name": f"Kelt Lower({self.kelt_ema})", "array": self._kelt_lower, "type": "overlay", "style": "dash", "color": "#26a69a"},
            {"name": f"ROC({self.roc_period})", "array": self._roc, "type": "subplot", "zero_line": True},
            {"name": "Chaikin Osc", "array": self._chaikin, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"Kelt Upper({self.kelt_ema})", datetimes, self._kelt_upper, style="dash", color="#ef5350"),
                self._make_overlay(f"Kelt Mid({self.kelt_ema})", datetimes, self._kelt_mid, color="#ffab40"),
                self._make_overlay(f"Kelt Lower({self.kelt_ema})", datetimes, self._kelt_lower, style="dash", color="#26a69a"),
            ],
            "subplots": [
                self._make_subplot(
                    f"ROC({self.roc_period})",
                    [self._make_subplot_trace("ROC", datetimes, self._roc, color="#bb86fc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "Chaikin Osc",
                    [self._make_subplot_trace("CO", datetimes, self._chaikin, color="#ef5350")],
                    zero_line=True,
                ),
            ],
        }
