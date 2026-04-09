"""MildTrendShortI1hV9 — DEMA slope + Elder Force negative + BSP sell dominant.

Economic logic: DEMA(20) declining slope on 1H provides double-smoothed low-lag
trend direction. Elder Force Index(10) < 0 validates volume-weighted selling
pressure. Buying/Selling Pressure ratio with sell > buy confirms seller dominance
in order flow. Signal smoothed with EMA(3) for mild-trend filtering.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.dema import dema
from indicators.momentum.elder_force import elder_force_index
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI1hV9(TrendingStrategy):
    """DEMA(20) slope < 0 + Elder Force(10) < 0 + BSP sell > buy."""

    name = "mild_trend_short_I_1h_v9"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 35

    dema_period: int = 20
    efi_period: int = 10
    bsp_period: int = 14
    chandelier_mult: float = 2.9

    _dema_line: np.ndarray | None = None
    _efi_arr: np.ndarray | None = None
    _buy_pressure: np.ndarray | None = None
    _sell_pressure: np.ndarray | None = None
    _pressure_ratio: np.ndarray | None = None
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
        self._dema_line = dema(self._closes, self.dema_period)
        self._efi_arr = elder_force_index(self._closes, self._volumes, self.efi_period)
        self._buy_pressure, self._sell_pressure, self._pressure_ratio = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes, self.bsp_period,
        )

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=3)

    def _raw_signal(self, bar_index: int) -> float:
        d = self._dema_line[bar_index]
        d_prev = self._dema_line[bar_index - 1]
        ef = self._efi_arr[bar_index]
        sp = self._sell_pressure[bar_index]
        bp = self._buy_pressure[bar_index]

        if any(np.isnan(v) for v in (d, d_prev, ef, sp, bp)):
            return 0.0

        slope = d - d_prev
        if slope >= 0:
            return 0.0

        signal = -0.3

        if ef < 0:
            signal -= 0.15

        if sp > bp and bp > 0:
            ratio = sp / bp
            signal -= min(0.2, (ratio - 1.0) * 0.3)

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"DEMA({self.dema_period})", "array": self._dema_line, "type": "overlay", "color": "#ab47bc"},
            {"name": f"Elder Force({self.efi_period})", "array": self._efi_arr,
             "type": "subplot", "zero_line": True},
            {"name": "Pressure Ratio", "array": self._pressure_ratio, "type": "subplot",
             "horizontal_lines": [1.0]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"DEMA({self.dema_period})", datetimes, self._dema_line, color="#ab47bc"),
            ],
            "subplots": [
                self._make_subplot(
                    f"Elder Force({self.efi_period})",
                    [self._make_subplot_trace("EFI", datetimes, self._efi_arr, color="#ef5350")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "BSP Ratio",
                    [self._make_subplot_trace("Ratio", datetimes, self._pressure_ratio, color="#ffab40")],
                    horizontal_lines=[1.0],
                ),
            ],
        }
