"""StrongTrendShortAG1hV9 — Keltner Breakdown + Williams%R Oversold + MFI Weak.

Economic logic: Price below Keltner lower (2.5x ATR) on 1H shows extreme
intraday weakness. Williams%R < -85 confirms bearish momentum continuing.
MFI < 30 shows money flow is extremely weak — strong short setup.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.williams_r import williams_r
from indicators.trend.keltner import keltner
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG1hV9(TrendingStrategy):
    """Keltner breakdown + Williams%R oversold + MFI weak.

    Signal logic:
        close < Kelt_lower AND WR < -90 AND MFI < 30 -> -0.85
        close < Kelt_mid AND WR < -80 -> -0.50
        else -> 0.0
    """

    name = "short_AG_1h_v9"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 35

    kelt_period: int = 25
    kelt_mult: float = 2.5
    wr_period: int = 18
    mfi_period: int = 20
    chandelier_mult: float = 3.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_period, atr_period=self.kelt_period,
            multiplier=self.kelt_mult,
        )
        self._wr = williams_r(
            self._highs, self._lows, self._closes, period=self.wr_period,
        )
        self._mfi = mfi(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.mfi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        kl = self._kelt_lower[bar_index]
        km = self._kelt_mid[bar_index]
        wr = self._wr[bar_index]
        m = self._mfi[bar_index]

        if np.isnan(c) or np.isnan(kl) or np.isnan(wr):
            return 0.0

        if c < kl and wr < -90 and (not np.isnan(m) and m < 30):
            return -0.85
        if c < km and wr < -80:
            return -0.50
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"Kelt Upper({self.kelt_period})", "array": self._kelt_upper, "type": "overlay", "style": "dash"},
            {"name": f"Kelt Mid({self.kelt_period})", "array": self._kelt_mid, "type": "overlay"},
            {"name": f"Kelt Lower({self.kelt_period})", "array": self._kelt_lower, "type": "overlay", "style": "dash"},
            {"name": f"Williams%R({self.wr_period})", "array": self._wr,
             "y_range": [-100, 0], "horizontal_lines": [-20, -80, -90]},
            {"name": f"MFI({self.mfi_period})", "array": self._mfi,
             "y_range": [0, 100], "horizontal_lines": [20, 30, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"Kelt Upper({self.kelt_period})", datetimes, self._kelt_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"Kelt Mid({self.kelt_period})", datetimes, self._kelt_mid, color="#ffab40"),
            self._make_overlay(f"Kelt Lower({self.kelt_period})", datetimes, self._kelt_lower, style="dash", color="#26a69a"),
        ]
        subplots = [
            self._make_subplot(f"Williams%R({self.wr_period})", [
                self._make_subplot_trace("Williams%R", datetimes, self._wr, color="#42a5f5"),
            ], horizontal_lines=[-20, -80, -90], y_range=[-100, 0]),
            self._make_subplot(f"MFI({self.mfi_period})", [
                self._make_subplot_trace("MFI", datetimes, self._mfi, color="#ab47bc"),
            ], horizontal_lines=[20, 30, 80], y_range=[0, 100]),
        ]
        return {"overlays": overlays, "subplots": subplots}
