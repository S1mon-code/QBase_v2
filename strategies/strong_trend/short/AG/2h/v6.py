"""StrongTrendShortAG2hV6 — LinReg Slope + Fisher Bearish + CMF Negative.

Economic logic: Linear Regression(35) slope < 0 captures the statistical
trend direction over ~3 days. Fisher Transform(18) < -0.8 confirms bearish
momentum in normalized price space. CMF(18) < -0.03 validates distribution
pressure. EMA(4) smoothing reduces whipsaw entries.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.ema import ema
from indicators.trend.linear_regression import linear_regression_slope
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendShortAG2hV6(TrendingStrategy):
    """LinReg(35) slope < 0 + Fisher(18) < -0.8 + CMF(18) < -0.03.

    Signal logic (raw, pre-smoothing):
        All 3 conditions met -> -0.90
        LinReg slope < 0 AND Fisher < -0.8 -> -0.55
        LinReg slope < 0 AND CMF < -0.03 -> -0.40
        else -> 0.0
    """

    name = "strong_trend_short_AG_2h_v6"
    horizon = "medium"
    direction = "short"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    lr_period: int = 35
    fisher_period: int = 18
    cmf_period: int = 18
    chandelier_mult: float = 3.2

    def on_init_arrays(self, closes, highs, lows, opens, volumes, oi, datetimes) -> None:
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._lr_slope = linear_regression_slope(self._closes, period=self.lr_period)
        self._fisher, self._fisher_trigger = fisher_transform(
            self._highs, self._lows, period=self.fisher_period,
        )
        self._cmf = cmf(self._highs, self._lows, self._closes, self._volumes,
                        period=self.cmf_period)

        n = len(closes)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(self.warmup, n):
            raw[i] = self._raw_signal(i)
        self._smooth_signal = ema(raw, period=4)

    def _raw_signal(self, i: int) -> float:
        sl = self._lr_slope[i]
        f = self._fisher[i]
        c = self._cmf[i]

        if any(np.isnan(v) for v in (sl, f, c)):
            return 0.0

        slope_neg = sl < 0.0
        fisher_bearish = f < -0.8
        cmf_neg = c < -0.03

        if slope_neg and fisher_bearish and cmf_neg:
            return -0.90
        if slope_neg and fisher_bearish:
            return -0.55
        if slope_neg and cmf_neg:
            return -0.40
        return 0.0

    def _generate_signal(self, bar_index: int) -> float:
        if bar_index < self.warmup:
            return 0.0
        val = self._smooth_signal[bar_index]
        return min(0.0, val) if not np.isnan(val) else 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"LinReg Slope({self.lr_period})", "array": self._lr_slope,
             "zero_line": True},
            {"name": f"Fisher({self.fisher_period})", "array": self._fisher,
             "panel": "Fisher", "zero_line": True, "horizontal_lines": [-0.8]},
            {"name": "Fisher Trigger", "array": self._fisher_trigger,
             "panel": "Fisher"},
            {"name": f"CMF({self.cmf_period})", "array": self._cmf, "zero_line": True},
        ]

    def get_indicator_panels(self, datetimes):
        subplots = [
            self._make_subplot(f"LinReg Slope({self.lr_period})", [
                self._make_subplot_trace("Slope", datetimes, self._lr_slope,
                                         color="#42a5f5"),
            ], zero_line=True),
            self._make_subplot("Fisher", [
                self._make_subplot_trace(f"Fisher({self.fisher_period})", datetimes,
                                         self._fisher, color="#ffab40"),
                self._make_subplot_trace("Trigger", datetimes, self._fisher_trigger,
                                         color="#ef5350"),
            ], zero_line=True, horizontal_lines=[-0.8]),
            self._make_subplot(f"CMF({self.cmf_period})", [
                self._make_subplot_trace("CMF", datetimes, self._cmf, color="#ab47bc"),
            ], zero_line=True),
        ]
        return {"overlays": [], "subplots": subplots}
