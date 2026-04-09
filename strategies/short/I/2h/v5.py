"""MildTrendShortI2hV5 — FRAMA Below Price + Ergodic Bearish + OI Momentum Contraction.

Economic logic: FRAMA adapts to market fractal dimension, becoming faster in trends.
Price below FRAMA with Ergodic oscillator negative confirms dual bearish momentum.
OI momentum contraction signals position unwinding.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.frama import frama
from indicators.momentum.ergodic import ergodic
from indicators.volume.oi_momentum import oi_momentum
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI2hV5(TrendingStrategy):
    """Price below FRAMA(50) + Ergodic(35,18)<0 + OI_Momentum(40)<0."""

    name = "short_I_2h_v5"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 65

    frama_period: int = 50
    ergo_short: int = 35
    ergo_long: int = 18
    oi_mom_period: int = 40
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
        self._frama = frama(self._closes, period=self.frama_period)
        self._ergo_line, self._ergo_sig = ergodic(
            self._closes, short_period=self.ergo_long, long_period=self.ergo_short,
        )
        self._oi_mom = oi_momentum(self._oi, self.oi_mom_period)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        fr = self._frama[bar_index]
        ergo = self._ergo_line[bar_index]
        oim = self._oi_mom[bar_index]

        if any(np.isnan(v) for v in (close, fr, ergo)):
            return 0.0

        if close >= fr:
            return 0.0

        dist = (fr - close) / fr
        strength = min(1.0, dist * 30.0)

        signal = -(0.25 + strength * 0.3)

        if ergo < 0:
            ergo_str = min(1.0, abs(ergo) / 30.0)
            signal -= 0.2 * ergo_str

        if not np.isnan(oim) and oim < 0:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"FRAMA({self.frama_period})", "array": self._frama, "type": "overlay", "color": "#ffab40"},
            {"name": "Ergodic", "array": self._ergo_line, "type": "subplot", "panel": "Ergodic", "zero_line": True},
            {"name": "Ergo Signal", "array": self._ergo_sig, "type": "subplot", "panel": "Ergodic", "style": "dash"},
            {"name": f"OI Mom({self.oi_mom_period})", "array": self._oi_mom, "type": "subplot", "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"FRAMA({self.frama_period})", datetimes, self._frama, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    "Ergodic",
                    [
                        self._make_subplot_trace("Ergodic", datetimes, self._ergo_line, color="#bb86fc"),
                        self._make_subplot_trace("Signal", datetimes, self._ergo_sig, style="dash", color="#78909c"),
                    ],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"OI Mom({self.oi_mom_period})",
                    [self._make_subplot_trace("OI Mom", datetimes, self._oi_mom, color="#ef5350")],
                    zero_line=True,
                ),
            ],
        }
