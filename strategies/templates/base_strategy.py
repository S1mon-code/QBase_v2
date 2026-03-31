"""Abstract base class for ALL QBase_v2 strategies.

# NOTE: See AlphaForge CLAUDE.md for BacktestContext API, BacktestConfig, etc.

Every strategy in the QBase_v2 system inherits from QBaseStrategy, which
enforces a uniform signal interface compatible with the Signal Blender and
the rest of the pipeline (directional filter, vol targeting, Chandelier Exit).

Subclasses must define class-level attributes and implement _generate_signal.
The strategy is deliberately thin: it produces a raw signal in [-1, +1] and
leaves position sizing, direction filtering, and execution to outer layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class QBaseStrategy(ABC):
    """Base class for all QBase_v2 strategies.

    Subclasses must define:
        name            -- Unique strategy identifier (e.g. "tsmom_fast").
        regime          -- "trending" or "mean_reversion".
        horizon         -- "fast", "medium", "slow", or None (for MR).
        signal_dimensions -- List of signal dimensions used,
                            e.g. ["momentum", "carry"].
        warmup          -- Minimum number of bars before a valid signal.

    Subclasses must implement:
        _generate_signal(bar_index) -> float  (-1.0 to +1.0)

    Optional overrides:
        on_init_arrays(closes, highs, lows, opens, volumes, oi, datetimes)
        get_indicator_config() -> list[dict]  (for attribution)
    """

    # --- Required class attributes (set by subclass) ---
    name: ClassVar[str]
    regime: ClassVar[str]
    horizon: ClassVar[str | None]
    signal_dimensions: ClassVar[list[str]]
    warmup: ClassVar[int]

    # --- Chandelier Exit parameter (optimisable) ---
    chandelier_mult: float = 2.5

    # --- OHLCV arrays populated by on_init_arrays ---
    _closes: np.ndarray | None = None
    _highs: np.ndarray | None = None
    _lows: np.ndarray | None = None
    _opens: np.ndarray | None = None
    _volumes: np.ndarray | None = None
    _oi: np.ndarray | None = None
    _datetimes: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
        """Receive precomputed OHLCV + OI + datetime arrays.

        Called once before any bar iteration.  Subclasses should override
        this (calling super first) to precompute indicator arrays that
        will be indexed inside ``_generate_signal``.

        Args:
            closes:    Array of close prices.
            highs:     Array of high prices.
            lows:      Array of low prices.
            opens:     Array of open prices.
            volumes:   Array of volume values.
            oi:        Array of open interest values.
            datetimes: Array of datetime values.
        """
        self._closes = closes.astype(np.float64)
        self._highs = highs.astype(np.float64)
        self._lows = lows.astype(np.float64)
        self._opens = opens.astype(np.float64)
        self._volumes = volumes.astype(np.float64)
        self._oi = oi.astype(np.float64)
        self._datetimes = datetimes

    # ------------------------------------------------------------------
    # Signal interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _generate_signal(self, bar_index: int) -> float:
        """Return a raw signal in [-1.0, +1.0] for the given bar.

        Positive values indicate a long bias, negative values a short bias.
        Zero means no signal.  During the warmup period the strategy should
        return 0.0 (or np.nan which the caller treats as 0).

        The strategy must NOT handle position sizing or direction filtering;
        those are applied by outer pipeline layers.

        Args:
            bar_index: Current bar index into the precomputed arrays.

        Returns:
            Signal strength from -1.0 (max short) to +1.0 (max long).
        """

    def generate_signals(self) -> np.ndarray:
        """Generate signals for all bars.

        Iterates over each bar, calling ``_generate_signal``.  Values
        during the warmup period are set to 0.0.  The result is clipped
        to [-1, 1].

        Returns:
            1-D float array of length ``len(self._closes)``.
        """
        if self._closes is None:
            raise RuntimeError(
                "on_init_arrays must be called before generate_signals"
            )

        n = len(self._closes)
        signals = np.zeros(n, dtype=np.float64)

        for i in range(self.warmup, n):
            raw = self._generate_signal(i)
            if np.isnan(raw):
                raw = 0.0
            signals[i] = np.clip(raw, -1.0, 1.0)

        return signals

    # ------------------------------------------------------------------
    # Attribution helper
    # ------------------------------------------------------------------

    def get_indicator_config(self) -> list[dict]:
        """Return indicator configuration for attribution analysis.

        Each dict should describe one indicator used by the strategy:
        ``{"name": "supertrend", "params": {"period": 10, "multiplier": 3.0}}``.

        Override in subclasses to provide strategy-specific attribution.
        """
        return []

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Enforce that concrete subclasses define required attributes."""
        super().__init_subclass__(**kwargs)
        # Skip enforcement on intermediate template classes
        if cls.__name__ in ("TrendingStrategy", "MeanReversionStrategy"):
            return
        required = ("name", "regime", "signal_dimensions", "warmup")
        for attr in required:
            if not hasattr(cls, attr) or getattr(cls, attr) is None:
                raise TypeError(
                    f"Concrete strategy {cls.__name__} must define "
                    f"class attribute '{attr}'"
                )
        if cls.regime not in ("trending", "mean_reversion"):
            raise TypeError(
                f"{cls.__name__}.regime must be 'trending' or "
                f"'mean_reversion', got '{cls.regime}'"
            )
        if cls.regime == "trending" and cls.horizon not in (
            "fast",
            "medium",
            "slow",
        ):
            raise TypeError(
                f"Trending strategy {cls.__name__} must set horizon to "
                f"'fast', 'medium', or 'slow', got '{cls.horizon}'"
            )

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"regime={self.regime!r} horizon={self.horizon!r}>"
        )
