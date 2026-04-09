"""MildTrendLongI1hV34 — Bollinger Band Breakout + Force Index Confirmation.

Economic logic: Bollinger Band breakouts signal volatility expansion, but many
are false breakouts that quickly revert.  Force Index combines price change
with volume — only breakouts backed by genuine volume-weighted momentum persist.
We profit from traders who short breakouts without checking volume commitment.
"""
from __future__ import annotations

import numpy as np

from indicators.volatility.bollinger import bollinger_bands
from indicators.volume.force_index import force_index
from strategies.templates.trending_template import TrendingStrategy


class MildTrendLongI1hV34(TrendingStrategy):
    """Bollinger Band breakout confirmed by Force Index direction.

    Signal logic:
        Close > BB_upper AND FI > 0 → +1.0
        Close < BB_lower AND FI < 0 → -1.0
        Inside bands: position relative to mid, ±0.5 if FI agrees, 0 if not.
    """

    name = "long_I_1h_v34"
    horizon = "medium"
    direction = "long"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 53  # bb_period + fi_period + 20

    # Optimizable parameters
    bb_period: int = 20
    bb_std: float = 2.0
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
        """Precompute Bollinger Bands and Force Index arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            self._closes, period=self.bb_period, num_std=self.bb_std,
        )
        self._fi = force_index(self._closes, self._volumes, period=self.fi_period)

    def _generate_signal(self, bar_index: int) -> float:
        """Return signal from BB position + Force Index agreement."""
        close = self._closes[bar_index]
        upper = self._bb_upper[bar_index]
        mid = self._bb_mid[bar_index]
        lower = self._bb_lower[bar_index]
        fi_val = self._fi[bar_index]

        if (
            np.isnan(close) or np.isnan(upper) or np.isnan(mid)
            or np.isnan(lower) or np.isnan(fi_val)
        ):
            return 0.0

        # Breakout above upper band with volume confirmation
        if close > upper and fi_val > 0.0:
            return 1.0
        # Breakout below lower band with volume confirmation
        if close < lower and fi_val < 0.0:
            return -1.0

        # Inside bands: position relative to midline
        band_width = upper - lower
        if band_width <= 0.0:
            return 0.0

        position = (close - mid) / (band_width / 2.0)  # -1..+1
        position = max(-1.0, min(1.0, position))

        # Scale by FI agreement: 0.5 if agree, 0.0 if disagree
        if (position > 0.0 and fi_val > 0.0) or (position < 0.0 and fi_val < 0.0):
            return position * 0.5
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {"name": "bollinger_bands", "params": {"period": self.bb_period, "num_std": self.bb_std}},
            {"name": "force_index", "params": {"period": self.fi_period}},
        ]

    def get_indicator_panels(self, datetimes):
        """Return overlay and subplot panel definitions for charting."""
        overlays = [
            self._make_overlay(f"BB Upper({self.bb_period})", datetimes, self._bb_upper, style="dash", color="#ef5350"),
            self._make_overlay(f"BB Mid({self.bb_period})", datetimes, self._bb_mid, color="#ffab40"),
            self._make_overlay(f"BB Lower({self.bb_period})", datetimes, self._bb_lower, style="dash", color="#26a69a")
        ]
        subplots = [
            self._make_subplot(
                f"Force Index({self.fi_period})",
                [self._make_subplot_trace("Force Index", datetimes, self._fi, color="#bb86fc")],
                zero_line=True,
            )
        ]
        return {"overlays": overlays, "subplots": subplots}

