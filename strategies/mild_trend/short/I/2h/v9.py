"""MildTrendShortI2hV9 — TEMA Below Price + Stochastic Bearish + MFI Weak.

Economic logic: Price below TEMA on 2H confirms triple-smoothed bearish trend.
Stochastic %K below %D with both below 50 validates bearish momentum alignment.
MFI below 50 signals weak money flow supporting the decline.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.tema import tema
from indicators.momentum.stochastic import stochastic
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV9(TrendingStrategy):
    """Price below TEMA(35) + Stoch(30,8)<50 + MFI(60)<50."""

    name = "mild_trend_short_I_2h_v9"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    tema_period: int = 35
    stoch_k: int = 30
    stoch_d: int = 8
    mfi_period: int = 60
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
        self._tema = tema(self._closes, self.tema_period)
        self._stoch_k, self._stoch_d = stochastic(
            self._highs, self._lows, self._closes, self.stoch_k, self.stoch_d,
        )
        self._mfi = mfi(
            self._highs, self._lows, self._closes, self._volumes, self.mfi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        t = self._tema[bar_index]
        sk = self._stoch_k[bar_index]
        sd = self._stoch_d[bar_index]
        mfi_val = self._mfi[bar_index]

        if any(np.isnan(v) for v in (close, t, sk, sd)):
            return 0.0

        if close >= t:
            return 0.0

        dist = (t - close) / t
        strength = min(1.0, dist * 35.0)

        signal = -(0.25 + strength * 0.3)

        if sk < sd and sk < 50:
            signal -= 0.2

        if not np.isnan(mfi_val) and mfi_val < 50:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"TEMA({self.tema_period})", "array": self._tema, "type": "overlay", "color": "#ffab40"},
            {"name": "%K", "array": self._stoch_k, "type": "subplot", "panel": "Stochastic",
             "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
            {"name": "%D", "array": self._stoch_d, "type": "subplot", "panel": "Stochastic", "style": "dash"},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"TEMA({self.tema_period})", datetimes, self._tema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Stochastic",
                    [
                        self._make_subplot_trace("%K", datetimes, self._stoch_k, color="#bb86fc"),
                        self._make_subplot_trace("%D", datetimes, self._stoch_d, style="dash", color="#78909c"),
                    ],
                    horizontal_lines=[20, 50, 80], y_range=[0, 100],
                ),
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#26a69a")],
                    horizontal_lines=[20, 50, 80], y_range=[0, 100],
                ),
            ],
        }
