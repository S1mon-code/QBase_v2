"""MildTrendLongIDailyV33 — SuperTrend Direction + OI Flow Sustained Confirmation.

Economic logic: SuperTrend provides an ATR-adaptive trend channel that
eliminates most whipsaws. OI flow rising above its signal line indicates
that open interest is accumulating in the direction of the trend, a sign
that new participants are entering — not just short-covering. When OI does
not confirm, a weaker partial signal is issued rather than no trade.
"""

from __future__ import annotations

import numpy as np

from indicators.trend.supertrend import supertrend
from indicators.volume.oi_flow import oi_flow
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongIDailyV33(TrendingStrategy):
    """SuperTrend direction scaled by OI flow confirmation.

    Signal logic:
        - ST direction +1 AND OI flow > OI signal: +1.0
        - ST direction -1 AND OI flow < OI signal: -1.0
        - ST direction +1 but OI not confirming: +0.4
        - ST direction -1 but OI not confirming: -0.4

    Attributes:
        st_period:       SuperTrend ATR period.
        st_mult:         SuperTrend ATR multiplier.
        oi_period:       OI flow EMA signal period.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "long_I_daily_v33"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 48  # st_period(14) + oi_period(14) + 20

    st_period: int = 14
    st_mult: float = 3.0
    oi_period: int = 14
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
        """Precompute SuperTrend direction and OI flow arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        _, self._st_direction = supertrend(
            self._highs,
            self._lows,
            self._closes,
            period=self.st_period,
            multiplier=self.st_mult,
        )
        self._oi_flow, self._oi_signal = oi_flow(
            self._closes,
            self._oi,
            self._volumes,
            period=self.oi_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal in [-1, 1] based on SuperTrend direction and OI flow."""
        direction = self._st_direction[bar_index]
        oi_fl = self._oi_flow[bar_index]
        oi_sig = self._oi_signal[bar_index]

        if np.isnan(direction) or np.isnan(oi_fl) or np.isnan(oi_sig):
            return 0.0

        base = float(direction)  # 1.0 or -1.0
        oi_ok = (base > 0 and oi_fl > oi_sig) or (base < 0 and oi_fl < oi_sig)
        return base if oi_ok else base * 0.4

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "supertrend",
                "params": {
                    "period": self.st_period,
                    "multiplier": self.st_mult,
                },
            },
            {"name": "oi_flow", "params": {"period": self.oi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        subplots = [
            self._make_subplot(
                f"OI Flow({self.oi_period})",
                [
                    self._make_subplot_trace("OI Flow", datetimes, self._oi_flow, color="#bb86fc"),
                    self._make_subplot_trace("OI Signal", datetimes, self._oi_signal, color="#4fc3f7"),
                ],
                zero_line=True,
            ),
            self._make_subplot(
                f"ST Dir({self.st_period})",
                [self._make_subplot_trace("Direction", datetimes, self._st_direction, style="step", color="#bb86fc")],
            )
        ]
        return {"overlays": [], "subplots": subplots}
