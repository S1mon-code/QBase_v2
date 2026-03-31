"""Trend Medium V6 — ADX + CMF (Trend Strength + Money Flow).

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: ADX measures trend strength regardless of direction — a rising
ADX means price is moving in a sustained, organized manner rather than
oscillating.  DI+ vs DI- settles direction: DI+ above DI- means buyers are
winning each bar's range expansion.  CMF (Chaikin Money Flow) adds a capital
confirmation layer — positive CMF means closing prices consistently finishing
in the upper half of the bar's range on heavy volume, evidence that
institutional money is being deployed in the bullish direction.  When all
three align (strong trend, bullish DI, positive CMF) conviction is highest.
ADX weak but direction and flow still aligned gives a half signal; ADX weak
and CMF contradicting gives nothing.

Who loses money: Mean-reversion traders fading a strong directional trend, and
trend followers who enter without confirming that real capital is behind the
move (DI crossover without CMF confirmation).
"""

from __future__ import annotations

import numpy as np

from strategies.templates.trending_template import TrendingStrategy
from indicators.trend.adx import adx_with_di
from indicators.volume.cmf import cmf


class TrendMediumV6(TrendingStrategy):
    """ADX trend strength + DI direction + CMF money-flow confirmation.

    Full signal when ADX is above threshold (strong trend), DI+ > DI- (bullish
    direction), and CMF > 0 (money flowing in).  Half signal when direction and
    at least one of strength or flow agrees.  Full bearish signal when all three
    flip bearish; half bearish otherwise.

    Attributes:
        adx_period:      Lookback period for ADX and DI calculation.
        adx_threshold:   Minimum ADX value to classify a trend as strong.
        cmf_period:      Lookback period for CMF calculation.
        chandelier_mult: Chandelier Exit multiplier (optimisable).
    """

    name = "trend_medium_v6"
    horizon = "medium"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 60

    adx_period: int = 25
    adx_threshold: float = 12.0
    cmf_period: int = 10
    chandelier_mult: float = 3.25

    # --- Precomputed arrays ---
    _adx: np.ndarray | None = None
    _plus_di: np.ndarray | None = None
    _minus_di: np.ndarray | None = None
    _cmf: np.ndarray | None = None

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
        """Precompute ADX/DI and CMF arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._adx, self._plus_di, self._minus_di = adx_with_di(
            self._highs, self._lows, self._closes, period=self.adx_period,
        )
        self._cmf = cmf(
            self._highs, self._lows, self._closes, self._volumes,
            period=self.cmf_period,
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Combine ADX strength, DI direction, and CMF money flow."""
        adx_val = self._adx[bar_index]
        plus = self._plus_di[bar_index]
        minus = self._minus_di[bar_index]
        cmf_val = self._cmf[bar_index]

        if np.isnan(adx_val) or np.isnan(plus) or np.isnan(minus) or np.isnan(cmf_val):
            return 0.0

        bull = plus > minus
        flow_bull = cmf_val > 0
        strong = adx_val > self.adx_threshold

        # Strongest bullish: trend confirmed by strength + direction + capital
        if bull and strong and flow_bull:
            return 1.0
        # Partial bullish: direction right, at least one confirming factor
        if bull and (strong or flow_bull):
            return 0.5
        # Strongest bearish: all three flip
        if not bull and not strong and not flow_bull:
            return -1.0
        # Partial bearish: direction wrong, at least one opposing factor
        if not bull:
            return -0.5
        return 0.0

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "adx_with_di",
                "params": {
                    "period": self.adx_period,
                    "threshold": self.adx_threshold,
                },
            },
            {
                "name": "cmf",
                "params": {"period": self.cmf_period},
            },
        ]
