"""StrongTrendShortAG4hV6 — ZLEMA Bearish + Williams%R Oversold + Smart Money.

Economic logic: ZLEMA reduces lag — price below ZLEMA30 on 4H is a strong bearish
signal. Williams%R below -80 confirms oversold momentum continuing down.
Smart Money Index declining shows institutions are distributing.
"""
from __future__ import annotations

import numpy as np

from indicators.momentum.williams_r import williams_r
from indicators.structure.smart_money import smart_money_index
from indicators.trend.zlema import zlema
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG4hV6(TrendingStrategy):
    """ZLEMA bearish + Williams%R oversold + SMI distribution.

    Signal logic:
        close < ZLEMA AND WR < -85 AND SMI declining -> -0.85
        close < ZLEMA AND WR < -80 -> -0.45
        else -> 0.0
    """

    name = "short_AG_4h_v6"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 70

    zlema_period: int = 30
    wr_period: int = 25
    smi_period: int = 50
    chandelier_mult: float = 3.5

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._zlema = zlema(self._closes, self.zlema_period)
        self._wr = williams_r(
            self._highs, self._lows, self._closes, period=self.wr_period,
        )
        self._smi, self._smi_signal = smart_money_index(
            self._opens, self._closes, self._highs, self._lows, self._volumes,
            period=self.smi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        z = self._zlema[bar_index]
        wr = self._wr[bar_index]
        smi = self._smi[bar_index]

        if np.isnan(c) or np.isnan(z) or np.isnan(wr):
            return 0.0

        if c >= z:
            return 0.0

        smi_declining = False
        if bar_index > 0 and not np.isnan(smi) and not np.isnan(self._smi[bar_index - 1]):
            smi_declining = smi < self._smi[bar_index - 1]

        if wr < -85 and smi_declining:
            return -0.85
        if wr < -80:
            return -0.45
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"ZLEMA({self.zlema_period})", "array": self._zlema, "type": "overlay"},
            {"name": f"Williams%R({self.wr_period})", "array": self._wr,
             "y_range": [-100, 0], "horizontal_lines": [-20, -80]},
            {"name": "SMI", "array": self._smi, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"ZLEMA({self.zlema_period})", datetimes, self._zlema, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot(f"Williams%R({self.wr_period})", [
                self._make_subplot_trace("Williams%R", datetimes, self._wr, color="#42a5f5"),
            ], horizontal_lines=[-20, -80], y_range=[-100, 0]),
            self._make_subplot("SMI", [
                self._make_subplot_trace("SMI", datetimes, self._smi, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
