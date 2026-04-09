"""StrongTrendShortAGDailyV9 — Keltner Breakdown + CCI Bearish + MFI Divergence.

Economic logic: Price below Keltner lower band (3.0x ATR) signals extreme bearish
conditions for volatile silver. CCI < -100 confirms oversold momentum but in a
strong trend context means continuation. MFI < 40 shows money is leaving.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.cci import cci
from indicators.trend.keltner import keltner
from indicators.volume.mfi import mfi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV9(TrendingStrategy):
    """Keltner breakdown + CCI bearish + MFI weak.

    Signal logic:
        close < Kelt_lower AND CCI < -100 AND MFI < 40 -> -0.90
        close < Kelt_mid AND CCI < -75 -> -0.45
        else -> 0.0
    """

    name = "strong_trend_short_AG_daily_v9"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 150

    kelt_period: int = 100
    kelt_mult: float = 3.0
    cci_period: int = 70
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            self._highs, self._lows, self._closes,
            ema_period=self.kelt_period, atr_period=self.kelt_period,
            multiplier=self.kelt_mult,
        )
        self._cci = cci(self._highs, self._lows, self._closes, period=self.cci_period)
        self._mfi = mfi(self._highs, self._lows, self._closes, self._volumes, period=40)

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        kl = self._kelt_lower[bar_index]
        km = self._kelt_mid[bar_index]
        cc = self._cci[bar_index]
        m = self._mfi[bar_index]

        if np.isnan(c) or np.isnan(kl) or np.isnan(cc):
            return 0.0

        if c < kl and cc < -100 and (not np.isnan(m) and m < 40):
            return -0.90
        if c < km and cc < -75:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"Keltner Upper({self.kelt_period})", "array": self._kelt_upper, "type": "overlay", "style": "dash"},
            {"name": f"Keltner Mid({self.kelt_period})", "array": self._kelt_mid, "type": "overlay"},
            {"name": f"Keltner Lower({self.kelt_period})", "array": self._kelt_lower, "type": "overlay", "style": "dash"},
            {"name": f"CCI({self.cci_period})", "array": self._cci, "zero_line": True, "horizontal_lines": [-100, 100]},
            {"name": "MFI", "array": self._mfi, "y_range": [0, 100], "horizontal_lines": [20, 40, 80]},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"Kelt Upper({self.kelt_period})", datetimes, self._kelt_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"Kelt Mid({self.kelt_period})", datetimes, self._kelt_mid, color="#ffab40"),
            self._make_overlay(f"Kelt Lower({self.kelt_period})", datetimes, self._kelt_lower, style="dash", color="#26a69a"),
        ]
        subplots = [
            self._make_subplot(f"CCI({self.cci_period})", [
                self._make_subplot_trace("CCI", datetimes, self._cci, color="#42a5f5"),
            ], zero_line=True, horizontal_lines=[-100, 100]),
            self._make_subplot("MFI", [
                self._make_subplot_trace("MFI", datetimes, self._mfi, color="#ab47bc"),
            ], horizontal_lines=[20, 40, 80], y_range=[0, 100]),
        ]
        return {"overlays": overlays, "subplots": subplots}
