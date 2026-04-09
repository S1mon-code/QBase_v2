"""MildTrendShortI4hV1 — ADX Trend + DI- Dominance + OBV Distribution.

Economic logic: ADX(22) > 18 confirms a trending regime on 4H Iron Ore.
DI-(22) > DI+ establishes bearish directional dominance. OBV falling below
its SMA(35) reveals persistent volume-weighted distribution, signaling
institutional selling pressure beneath the surface.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.adx import adx_with_di
from indicators.volume.obv import obv
from indicators.trend.sma import sma
from indicators.trend.ema import ema
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV1(TrendingStrategy):
    """ADX(22) > 18 + DI-(22) > DI+ + OBV < OBV_SMA(35)."""

    name = "short_I_4h_v1"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    adx_period: int = 22
    obv_sma_period: int = 35
    chandelier_mult: float = 3.4

    # -- precomputed arrays --
    _adx_arr: np.ndarray | None = None
    _plus_di: np.ndarray | None = None
    _minus_di: np.ndarray | None = None
    _obv_arr: np.ndarray | None = None
    _obv_sma: np.ndarray | None = None
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
        self._obv_arr = obv(self._closes, self._volumes)
        self._obv_sma = sma(self._obv_arr, self.obv_sma_period)

        n = len(closes)
        raw = np.zeros(n)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, bar_index: int) -> float:
        adx_val = self._adx_arr[bar_index]
        pdi = self._plus_di[bar_index]
        mdi = self._minus_di[bar_index]
        obv_val = self._obv_arr[bar_index]
        obv_sma_val = self._obv_sma[bar_index]

        if any(np.isnan(v) for v in (adx_val, pdi, mdi, obv_sma_val)):
            return 0.0

        if adx_val <= 18 or mdi <= pdi:
            return 0.0

        # Base signal from DI spread
        di_spread = (mdi - pdi) / max(mdi + pdi, 1e-9)
        strength = min(1.0, di_spread * 3.0)
        signal = -(0.3 + strength * 0.2)

        # ADX strength bonus
        if adx_val > 25:
            signal -= 0.1

        # OBV distribution confirmation
        if obv_val < obv_sma_val:
            signal -= 0.15

        return max(-0.65, signal)

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"ADX({self.adx_period})", "array": self._adx_arr,
             "type": "subplot", "panel": "ADX", "y_range": [0, 100],
             "horizontal_lines": [18, 25]},
            {"name": "+DI", "array": self._plus_di, "type": "subplot",
             "panel": "ADX", "color": "#26a69a"},
            {"name": "-DI", "array": self._minus_di, "type": "subplot",
             "panel": "ADX", "color": "#ef5350"},
            {"name": "OBV", "array": self._obv_arr, "type": "subplot",
             "panel": "OBV", "color": "#42a5f5"},
            {"name": f"OBV SMA({self.obv_sma_period})", "array": self._obv_sma,
             "type": "subplot", "panel": "OBV", "color": "#ffab40"},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [],
            "subplots": [
                self._make_subplot(
                    "ADX",
                    [
                        self._make_subplot_trace("ADX", datetimes, self._adx_arr, color="#bb86fc"),
                        self._make_subplot_trace("+DI", datetimes, self._plus_di, color="#26a69a"),
                        self._make_subplot_trace("-DI", datetimes, self._minus_di, color="#ef5350"),
                    ],
                    y_range=[0, 100], horizontal_lines=[18, 25],
                ),
                self._make_subplot(
                    "OBV",
                    [
                        self._make_subplot_trace("OBV", datetimes, self._obv_arr, color="#42a5f5"),
                        self._make_subplot_trace(f"SMA({self.obv_sma_period})", datetimes, self._obv_sma, color="#ffab40"),
                    ],
                ),
            ],
        }
