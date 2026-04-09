"""StrongTrendShortAGDailyV5 — Linear Regression Bearish + Vortex Bear + Smart Money Sell.

Economic logic: Linear regression slope below price confirms downtrend.
Vortex VI- > VI+ shows bearish pressure dominates. Smart Money Index declining
signals institutional distribution.
"""
from __future__ import annotations

import numpy as np

from indicators.structure.smart_money import smart_money_index
from indicators.trend.linear_regression import linear_regression
from indicators.trend.vortex import vortex
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAGDailyV5(TrendingStrategy):
    """LinReg bearish + Vortex bear + Smart Money distribution.

    Signal logic:
        close < LinReg AND VI_minus > VI_plus AND SMI declining -> -0.85
        close < LinReg AND VI_minus > VI_plus -> -0.50
        else -> 0.0
    """

    name = "short_AG_daily_v5"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 250

    lr_period: int = 200
    vortex_period: int = 80
    smi_period: int = 150
    chandelier_mult: float = 4.0

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._lr = linear_regression(self._closes, period=self.lr_period)
        self._vi_plus, self._vi_minus = vortex(
            self._highs, self._lows, self._closes, period=self.vortex_period,
        )
        self._smi, self._smi_signal = smart_money_index(
            self._opens, self._closes, self._highs, self._lows, self._volumes,
            period=self.smi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        c = self._closes[bar_index]
        lr = self._lr[bar_index]
        vp = self._vi_plus[bar_index]
        vm = self._vi_minus[bar_index]
        smi = self._smi[bar_index]

        if np.isnan(c) or np.isnan(lr) or np.isnan(vp) or np.isnan(vm):
            return 0.0

        if c >= lr:
            return 0.0

        smi_declining = False
        if bar_index > 0 and not np.isnan(smi) and not np.isnan(self._smi[bar_index - 1]):
            smi_declining = smi < self._smi[bar_index - 1]

        if vm > vp and smi_declining:
            return -0.85
        if vm > vp:
            return -0.50
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"LinReg({self.lr_period})", "array": self._lr, "type": "overlay"},
            {"name": "VI+", "array": self._vi_plus, "panel": "Vortex"},
            {"name": "VI-", "array": self._vi_minus, "panel": "Vortex"},
            {"name": "SMI", "array": self._smi, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay(f"LinReg({self.lr_period})", datetimes, self._lr, color="#ff7043"),
        ]
        subplots = [
            self._make_subplot("Vortex", [
                self._make_subplot_trace("VI+", datetimes, self._vi_plus, color="#26a69a"),
                self._make_subplot_trace("VI-", datetimes, self._vi_minus, color="#ef5350"),
            ], horizontal_lines=[1.0]),
            self._make_subplot("SMI", [
                self._make_subplot_trace("SMI", datetimes, self._smi, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": overlays, "subplots": subplots}
