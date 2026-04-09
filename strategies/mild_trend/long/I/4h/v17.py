"""MildTrendLongI4hV17 — Linear Regression Slope + Force Index.

Economic logic: Linear regression slope captures the underlying rate of price change,
normalized by ATR for cross-regime comparability.  Force Index confirms that volume
momentum supports the regression direction.  When both slope and force agree, the
trend has both statistical and volume backing.
"""
from __future__ import annotations

import numpy as np

from indicators.trend.linear_regression import linear_regression
from indicators.volatility.atr import atr
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI4hV17(TrendingStrategy):
    """Linear regression slope normalized by ATR, confirmed by Force Index.

    Signal logic:
        slope > 0 AND FI > 0 → +min(1.0, norm_slope * 5)
        slope < 0 AND FI < 0 → -min(1.0, abs(norm_slope) * 5)
        Disagree → 0.0
    """

    name = "mild_trend_long_I_4h_v17"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 53  # lr_period + fi_period + 20

    # Optimizable parameters
    lr_period: int = 20
    fi_period: int = 13
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
        """Precompute linear regression, ATR, and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._lr = linear_regression(self._closes, period=self.lr_period)
        self._atr = atr(self._highs, self._lows, self._closes, period=self.lr_period)
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from LR slope direction filtered by Force Index."""
        if bar_index < 1:
            return 0.0

        lr_cur = self._lr[bar_index]
        lr_prev = self._lr[bar_index - 1]
        atr_val = self._atr[bar_index]
        fi_val = self._fi[bar_index]

        if np.isnan(lr_cur) or np.isnan(lr_prev) or np.isnan(atr_val) or np.isnan(fi_val):
            return 0.0

        if atr_val <= 0.0:
            return 0.0

        slope = lr_cur - lr_prev
        norm_slope = slope / atr_val

        if norm_slope > 0.0 and fi_val > 0.0:
            return min(1.0, norm_slope * 5.0)
        if norm_slope < 0.0 and fi_val < 0.0:
            return -min(1.0, abs(norm_slope) * 5.0)

        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "linear_regression", "params": {"period": self.lr_period}},
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]
