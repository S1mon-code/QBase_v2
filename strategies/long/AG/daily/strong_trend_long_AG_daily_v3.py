"""StrongTrendLongAGDailyV3 — SuperTrend(100,4.0) + TSI(70,35) + Volume-Weighted RSI(100).

Economic logic: SuperTrend with wide multiplier (4.0) captures Silver's high
volatility daily trends without false breakouts. TSI double-smoothing filters
noise from AG's choppy momentum. Volume-weighted RSI confirms that buying
pressure accompanies upward momentum.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.momentum.tsi import tsi
from indicators.volume.volume_weighted_rsi import volume_rsi
from strategies.templates.trending_template import TrendingStrategy


class StrongTrendLongAGDailyV3(TrendingStrategy):
    """SuperTrend direction + TSI momentum + Volume-weighted RSI confirmation.

    Signal logic:
        - SuperTrend bullish AND TSI > 0 AND VWRSI > 50 → long signal
        - Strength scales with TSI magnitude and VWRSI distance from 50

    Attributes:
        st_period:        SuperTrend ATR period.
        st_mult:          SuperTrend multiplier (wide for AG volatility).
        tsi_long:         TSI long smoothing period.
        tsi_short:        TSI short smoothing period.
        chandelier_mult:  Chandelier Exit multiplier (optimisable).
    """

    name = "long_AG_daily_v3"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 200

    st_period: int = 100
    st_mult: float = 4.0
    tsi_long: int = 70
    tsi_short: int = 35
    chandelier_mult: float = 4.0

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
        self._st_line, self._st_dir = supertrend(
            self._highs, self._lows, self._closes,
            period=self.st_period, multiplier=self.st_mult,
        )
        self._tsi_line, self._tsi_signal = tsi(
            self._closes, long_period=self.tsi_long, short_period=self.tsi_short,
        )
        self._vwrsi = volume_rsi(self._closes, self._volumes, period=100)

    def _generate_signal(self, bar_index: int) -> float:
        st_d = self._st_dir[bar_index]
        tsi_val = self._tsi_line[bar_index]
        vwrsi = self._vwrsi[bar_index]

        if np.isnan(st_d) or np.isnan(tsi_val) or np.isnan(vwrsi):
            return 0.0

        st_bull = st_d > 0
        tsi_bull = tsi_val > 0.0
        vwrsi_bull = vwrsi > 50.0

        if st_bull and tsi_bull and vwrsi_bull:
            strength = min(1.0, tsi_val / 30.0 * 0.5 + (vwrsi - 50.0) / 50.0 * 0.5)
            return max(0.0, strength)
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        return [
            {"name": f"SuperTrend({self.st_period},{self.st_mult})",
             "array": self._st_line, "type": "overlay"},
            {"name": f"TSI({self.tsi_long},{self.tsi_short})",
             "array": self._tsi_line, "type": "subplot", "zero_line": True},
            {"name": "VWRSI(100)", "array": self._vwrsi, "type": "subplot",
             "y_range": [0, 100], "horizontal_lines": [30, 50, 70]},
        ]

    def get_indicator_panels(self, datetimes):
        return {
            "overlays": [
                self._make_overlay(f"SuperTrend({self.st_period},{self.st_mult})",
                                   datetimes, self._st_line, color="#ff7043"),
            ],
            "subplots": [
                self._make_subplot(
                    f"TSI({self.tsi_long},{self.tsi_short})",
                    [self._make_subplot_trace("TSI", datetimes, self._tsi_line, color="#ab47bc"),
                     self._make_subplot_trace("Signal", datetimes, self._tsi_signal, color="#ff8a65")],
                    zero_line=True,
                ),
                self._make_subplot(
                    "VWRSI(100)",
                    [self._make_subplot_trace("VWRSI", datetimes, self._vwrsi, color="#42a5f5")],
                    y_range=[0, 100], horizontal_lines=[30, 50, 70],
                ),
            ],
        }
