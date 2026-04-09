"""MildTrendShortI4hV6 — ZLEMA Below Price + CCI Bearish + MFI Weak.

Economic logic: ZLEMA's zero-lag property reacts quickly to price changes on 4H.
Price below ZLEMA with CCI below -50 confirms weak momentum. MFI below 50
validates that money flow supports the bearish thesis.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.zlema import zlema
from indicators.momentum.cci import cci
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class MildTrendShortI4hV6(TrendingStrategy):
    """Price below ZLEMA(30) + CCI(30)<-50 + MFI(50)<50."""

    name = "mild_trend_short_I_4h_v6"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    zlema_period: int = 30
    cci_period: int = 30
    mfi_period: int = 50
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
        self._zlema = zlema(self._closes, self.zlema_period)
        self._cci = cci(self._highs, self._lows, self._closes, self.cci_period)
        self._mfi = mfi(
            self._highs, self._lows, self._closes, self._volumes, self.mfi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        close = self._closes[bar_index]
        zl = self._zlema[bar_index]
        cci_val = self._cci[bar_index]
        mfi_val = self._mfi[bar_index]

        if any(np.isnan(v) for v in (close, zl, cci_val)):
            return 0.0

        if close >= zl:
            return 0.0

        dist = (zl - close) / zl
        strength = min(1.0, dist * 35.0)

        signal = -(0.25 + strength * 0.3)

        if cci_val < -50:
            cci_str = min(1.0, abs(cci_val + 50) / 150.0)
            signal -= 0.2 * cci_str

        if not np.isnan(mfi_val) and mfi_val < 50:
            signal -= 0.15

        return max(-1.0, signal)

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"ZLEMA({self.zlema_period})", "array": self._zlema, "type": "overlay", "color": "#ffab40"},
            {"name": f"CCI({self.cci_period})", "array": self._cci, "type": "subplot", "zero_line": True},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
        ]

    def get_indicator_panels(self, datetimes: np.ndarray) -> dict:
        return {
            "overlays": [
                self._make_overlay(f"ZLEMA({self.zlema_period})", datetimes, self._zlema, color="#ffab40"),
            ],
            "subplots": [
                self._make_subplot(
                    f"CCI({self.cci_period})",
                    [self._make_subplot_trace("CCI", datetimes, self._cci, color="#bb86fc")],
                    zero_line=True,
                ),
                self._make_subplot(
                    f"MFI({self.mfi_period})",
                    [self._make_subplot_trace("MFI", datetimes, self._mfi, color="#26a69a")],
                    horizontal_lines=[20, 50, 80], y_range=[0, 100],
                ),
            ],
        }
