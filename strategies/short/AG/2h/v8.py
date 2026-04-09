"""StrongTrendShortAG2hV8 — PSAR Bearish + Elder Force Negative + BSP Sell Dominant.

Economic logic: Parabolic SAR bearish direction (-1) confirms price is below
the trailing stop reference — trend is down. Elder Force Index(15) < 0 validates
bearish volume-price force. Buying/Selling Pressure ratio with sell > buy * 1.1
confirms sellers dominate. EMA(4) smoothing prevents overtrading.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.psar import psar
from indicators.momentum.elder_force import elder_force_index
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV8(TrendingStrategy):
    """PSAR bearish + Elder Force(15) < 0 + BSP sell > buy * 1.1.

    Signal logic (raw, pre-smoothing):
        All 3 conditions met -> -0.90
        PSAR bearish AND Elder Force < 0 -> -0.55
        PSAR bearish AND BSP sell dominant -> -0.40
        else -> 0.0
    """

    name = "short_AG_2h_v8"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 55

    fi_period: int = 15
    bsp_period: int = 14
    chandelier_mult: float = 3.6

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._psar_vals, self._psar_dir = psar(self._highs, self._lows)
        self._elder_fi = elder_force_index(
            self._closes, self._volumes, period=self.fi_period,
        )
        self._buy_p, self._sell_p, self._pressure_ratio = buying_selling_pressure(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.bsp_period,
        )

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        pd = self._psar_dir[i]
        ef = self._elder_fi[i]
        bp = self._buy_p[i]
        sp = self._sell_p[i]

        if any(np.isnan(v) for v in (pd, ef, bp, sp)):
            return 0.0

        psar_bearish = pd < 0
        force_neg = ef < 0.0
        sell_dominant = sp > bp * 1.1 if bp > 0 else sp > 0

        if psar_bearish and force_neg and sell_dominant:
            return -0.90
        if psar_bearish and force_neg:
            return -0.55
        if psar_bearish and sell_dominant:
            return -0.40
        return 0.0

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": "PSAR", "array": self._psar_vals,
             "type": "overlay", "style": "step"},
            {"name": f"Elder FI({self.fi_period})", "array": self._elder_fi,
             "zero_line": True},
            {"name": "Buy Pressure", "array": self._buy_p,
             "panel": "BSP"},
            {"name": "Sell Pressure", "array": self._sell_p,
             "panel": "BSP"},
        ]

    def get_indicator_panels(self, datetimes):
        overlays = [
            self._make_overlay("PSAR", datetimes, self._psar_vals,
                               style="step", color="#ef5350"),
        ]
        subplots = [
            self._make_subplot(f"Elder FI({self.fi_period})", [
                self._make_subplot_trace("Elder FI", datetimes, self._elder_fi,
                                         color="#42a5f5"),
            ], zero_line=True),
            self._make_subplot("Buy/Sell Pressure", [
                self._make_subplot_trace("Buy", datetimes, self._buy_p,
                                         color="#66bb6a"),
                self._make_subplot_trace("Sell", datetimes, self._sell_p,
                                         color="#ef5350"),
            ]),
        ]
        return {"overlays": overlays, "subplots": subplots}
