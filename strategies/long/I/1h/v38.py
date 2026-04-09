"""MildTrendLongI1hV38 — Linear Regression Slope + CMF.

Economic logic: Linear regression slope quantifies the underlying price trend
rate, stripping out noise.  Normalizing by ATR makes the slope comparable
across different volatility regimes.  CMF confirms that money is flowing in
the direction of the regression slope.  Traders who trade against a
statistically significant slope with volume confirmation bleed steadily.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression
from indicators.volatility.atr import atr
from indicators.volume.cmf import cmf
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV38(TrendingStrategy):
    """Linear regression slope (ATR-normalized) + CMF agreement.

    Signal logic:
        slope_norm > 0 AND CMF > 0 → +min(1.0, slope_norm)
        slope_norm < 0 AND CMF < 0 → -min(1.0, |slope_norm|)
        Disagree → 0.0
    """

    name = "long_I_1h_v38"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 40  # max(lr_period, cmf_period) + 20

    # Optimizable parameters
    lr_period: int = 20
    cmf_period: int = 20
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
        """Precompute linear regression, ATR, and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._lr = linear_regression(self._closes, period=self.lr_period)
        self._atr = atr(self._highs, self._lows, self._closes, period=self.lr_period)
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from LR slope direction + CMF agreement."""
        lr_val = self._lr[bar_index]
        lr_prev = self._lr[bar_index - 1]
        atr_val = self._atr[bar_index]
        cmf_val = self._cmf[bar_index]

        if (
            np.isnan(lr_val) or np.isnan(lr_prev)
            or np.isnan(atr_val) or np.isnan(cmf_val)
            or atr_val <= 0.0
        ):
            return 0.0

        # Slope = change in LR value, normalized by ATR
        slope = lr_val - lr_prev
        slope_norm = slope / atr_val
        slope_norm = max(-1.0, min(1.0, slope_norm))

        # Require CMF agreement
        if slope_norm > 0.0 and cmf_val > 0.0:
            return min(1.0, slope_norm)
        if slope_norm < 0.0 and cmf_val < 0.0:
            return -min(1.0, abs(slope_norm))

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "linear_regression", "params": {"period": self.lr_period}},
            {"name": "cmf", "params": {"period": self.cmf_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"LinReg({self.lr_period})", datetimes, self._lr, color="#ffab40")
        ]
        subplots = [
            self._make_subplot(
                f"CMF({self.cmf_period})",
                [self._make_subplot_trace("CMF", datetimes, self._cmf, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}

