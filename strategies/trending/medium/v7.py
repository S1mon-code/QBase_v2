"""Trend Medium V7 — Vortex + Klinger Volume Oscillator.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Economic logic: Vortex Indicator (VI+ vs VI-) captures price structure
direction by measuring positive and negative trend movement over a rolling
window — VI+ dominating means buyers are driving new highs above prior lows,
signalling a structurally bullish market.  The Klinger Volume Oscillator (KVO
vs its signal line) maps the direction of capital flow: when fast money (short
EMA of volume * price force) outpaces slow money, the underlying trend has
institutional volume momentum behind it.  When both indicators agree — price
structure bullish AND capital flow accelerating upward — the signal is at full
conviction.  Partial agreement yields a reduced signal; full disagreement flips
to the short side.

Who loses money: Traders who follow price direction without verifying that
volume is confirming the move.  Vortex alone fires on structural breakouts that
lack participation (thin rallies, gap-and-fade); Klinger alone fires on volume
surges that are not yet reflected in price structure (distribution phases).
Fading traders who short a Vortex bull signal while KVO is still rising also
lose money because the two-dimensional filter catches those premature reversals.
"""

from __future__ import annotations

import numpy as np

from strategies.templates.trending_template import TrendingStrategy
from indicators.trend.vortex import vortex
from indicators.volume.klinger import klinger


class TrendMediumV7(TrendingStrategy):
    """Vortex price-structure direction + Klinger volume-momentum confirmation.

    Full bull signal when VI+ > VI- and KVO > signal line (both confirm long).
    Half bull signal when VI+ > VI- but KVO <= signal line (structure without
    volume momentum).  Full bear signal when VI- >= VI+ and KVO <= signal line
    (both confirm short).  Half bear signal when VI- >= VI+ but KVO > signal
    line (structure points short, volume momentum not yet aligned).

    Attributes:
        vortex_period:   Lookback period for the Vortex Indicator.
        klinger_fast:    Fast EMA period for the Klinger KVO.
        klinger_slow:    Slow EMA period for the Klinger KVO.
        chandelier_mult: Chandelier Exit ATR multiplier (optimisable).
    """

    name = "trend_medium_v7"
    horizon = "medium"
    signal_dimensions = ["momentum", "volume"]
    warmup: int = 100

    vortex_period: int = 23
    klinger_fast: int = 39
    klinger_slow: int = 65
    chandelier_mult: float = 2.25

    # --- Precomputed arrays ---
    _vi_plus: np.ndarray | None = None
    _vi_minus: np.ndarray | None = None
    _kvo: np.ndarray | None = None
    _kvo_signal: np.ndarray | None = None

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
        """Precompute Vortex and Klinger Volume Oscillator arrays."""
        super().on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)

        self._vi_plus, self._vi_minus = vortex(
            highs, lows, closes, self.vortex_period
        )
        self._kvo, self._kvo_signal = klinger(
            highs, lows, closes, volumes, self.klinger_fast, self.klinger_slow
        )

    def _generate_signal(self, bar_index: int) -> float:
        """Combine Vortex direction with Klinger volume momentum confirmation."""
        vip = self._vi_plus[bar_index]
        vim = self._vi_minus[bar_index]
        kvo = self._kvo[bar_index]
        sig = self._kvo_signal[bar_index]

        if np.isnan(vip) or np.isnan(vim) or np.isnan(kvo) or np.isnan(sig):
            return 0.0

        bull_vortex = vip > vim       # 价格结构看多
        bull_klinger = kvo > sig      # 量能看多

        if bull_vortex and bull_klinger:
            return 1.0    # 双重共振看多 — 最强做多信号
        if bull_vortex and not bull_klinger:
            return 0.5    # 价格结构看多，量能未确认 — 减仓做多
        if not bull_vortex and not bull_klinger:
            return -1.0   # 双重共振看空 — 最强做空信号
        return -0.5       # 价格结构看空，量能未确认 — 减仓做空

    def get_indicator_config(self) -> list[dict]:
        """Return indicator metadata for attribution."""
        return [
            {
                "name": "vortex",
                "params": {
                    "period": self.vortex_period,
                },
            },
            {
                "name": "klinger",
                "params": {
                    "fast": self.klinger_fast,
                    "slow": self.klinger_slow,
                },
            },
        ]
