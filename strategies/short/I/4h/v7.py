"""MildTrendShortI4hV7 — Donchian Lower Break + TSI Bearish + Volume Spike.

Economic logic: Breaking below Donchian lower band on 4H signals a new period low.
TSI below zero confirms double-smoothed momentum is bearish. Volume spikes
during the breakdown validate institutional participation in selling.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.donchian import donchian
from indicators.momentum.tsi import tsi
from indicators.volume.volume_spike import volume_spike
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV7(TrendingStrategy):
    """Break Donchian(40) lower + TSI(25,12)<0 + VolumeSpike(25)."""

    name = "short_I_4h_v7"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 55

    dc_period: int = 40
    tsi_long: int = 25
    tsi_short: int = 12
    vs_period: int = 25
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
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(
            self._highs, self._lows, self.dc_period,
        )
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, self.tsi_long, self.tsi_short,
        )
        self._vol_spike = volume_spike(self._volumes, self.vs_period).astype(np.float64)

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        dc_low = self._dc_lower[bar_index]
        tsi_val = self._tsi_line[bar_index]
        vs = self._vol_spike[bar_index]

        if any(np.isnan(v) for v in (close, dc_low, tsi_val)):
            return 0.0

        if close >= dc_low:
            return 0.0

        signal = -0.4

        if tsi_val < 0:
            tsi_str = min(1.0, abs(tsi_val) / 30.0)
            signal -= 0.25 * tsi_str

        if vs > 0.5:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"DC Upper({self.dc_period})", "array": self._dc_upper, "type": "overlay", "style": "dash", "color": "#ef5350"},
            {"name": f"DC Lower({self.dc_period})", "array": self._dc_lower, "type": "overlay", "style": "dash", "color": "#26a69a"},
            {"name": "TSI", "array": self._tsi_line, "type": "subplot", "panel": "TSI", "zero_line": True},
            {"name": "TSI Signal", "array": self._tsi_signal, "type": "subplot", "panel": "TSI", "style": "dash"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"DC Upper({self.dc_period})", datetimes, self._dc_upper, style="dash", color="#ef5350"),
                self._make_overlay(f"DC Lower({self.dc_period})", datetimes, self._dc_lower, style="dash", color="#26a69a"),
            ],
            "subplots": [
                self._make_subplot(
                    "TSI",
                    [
                        self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#bb86fc"),
                        self._make_subplot_trace("Signal", datetimes, self._tsi_signal, style="dash", color="#78909c"),
                    ],
                    zero_line=True,
                ),
            ],
        }
