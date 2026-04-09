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
        name            -- Unique strategy identifier (e.g. "long_I_daily_v1").
        regime          -- "trending" or "mean_reversion".
        horizon         -- "fast", "medium", "slow", or None (for MR).
        direction       -- "long", "short", or "both".
                           "long"  → signals clipped to [0, +1] (long only).
                           "short" → signals clipped to [-1, 0] (short only).
                           "both"  → signals in [-1, +1] (bidirectional, for MR).
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
    direction: ClassVar[str]  # "long" | "short" | "both"
    signal_dimensions: ClassVar[list[str]]
    warmup: ClassVar[int]

    # Valid direction values
    _VALID_DIRECTIONS: ClassVar[frozenset[str]] = frozenset({"long", "short", "both"})

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

        # Direction-based signal bounds
        if self.direction == "long":
            lo, hi = 0.0, 1.0
        elif self.direction == "short":
            lo, hi = -1.0, 0.0
        else:  # "both"
            lo, hi = -1.0, 1.0

        for i in range(self.warmup, n):
            raw = self._generate_signal(i)
            if np.isnan(raw):
                raw = 0.0
            signals[i] = np.clip(raw, lo, hi)

        return signals

    # ------------------------------------------------------------------
    # Attribution helper
    # ------------------------------------------------------------------

    def get_indicator_config(self) -> list[dict]:
        """Return indicator configuration for attribution and visualization.

        Each dict describes one indicator used by the strategy.  Two formats
        are supported:

        **Legacy (attribution only):**
        ``{"name": "supertrend", "params": {"period": 10, "multiplier": 3.0}}``

        **Extended (attribution + auto panel generation):**
        ``{"name": "EMA(20)", "array": self._ema, "type": "overlay"}``

        Extended format fields:
            name   -- Display name for the chart legend.
            array  -- numpy array of indicator values (same length as bars).
            type   -- "overlay" or "subplot".  If omitted, auto-classified
                      from indicator category (see _OVERLAY_INDICATORS).
            style  -- "line", "step", "dash", "bar", "area".  Default "line".
            color  -- Hex color string.  Auto-assigned if omitted.
            params -- Parameter dict (optional, for attribution).
            panel  -- Panel name to group multiple traces (e.g. "MACD").
                      Traces sharing the same panel name render together.
            y_range         -- Fixed Y-axis range, e.g. [0, 100].
            horizontal_lines -- Reference lines, e.g. [30, 70].
            zero_line        -- Draw zero-axis line (bool).
            color_positive   -- Bar positive color (bar style only).
            color_negative   -- Bar negative color (bar style only).

        When ``array`` is present, ``get_indicator_panels`` auto-generates
        panels without needing a manual override.
        """
        return []

    # ------------------------------------------------------------------
    # Indicator panel visualization (AlphaForge report integration)
    # ------------------------------------------------------------------

    # Indicators whose category is "trend" that produce price-level values.
    # These are rendered as overlays on the main K-line chart.
    # Everything else (momentum, volume, volatility, etc.) → subplot.
    _OVERLAY_INDICATORS: ClassVar[frozenset[str]] = frozenset({
        # trend/ directory — price-level moving averages and channels
        "ema", "sma", "dema", "tema", "alma", "frama", "hma", "kama",
        "mesa_adaptive_ma", "mama", "fama", "t3", "vwma", "zlema",
        "ema_ribbon", "higher_low",
        # trend/ — price-level bands and channels
        "supertrend", "donchian", "parabolic_sar", "ichimoku",
        "linear_regression_channel",
        # volatility/ — price-level bands
        "bollinger", "bollinger_bands", "keltner", "keltner_channel",
        "acceleration_bands", "chandelier_exit",
        # volume/ — price-level reference
        "vwap", "poc", "volume_profile",
    })

    # Indicators with known display hints (y_range, horizontal_lines, etc.)
    _SUBPLOT_HINTS: ClassVar[dict[str, dict]] = {
        "rsi": {"y_range": [0, 100], "horizontal_lines": [30, 70]},
        "mfi": {"y_range": [0, 100], "horizontal_lines": [20, 50, 80]},
        "stochastic": {"y_range": [0, 100], "horizontal_lines": [20, 80]},
        "williams_r": {"y_range": [-100, 0], "horizontal_lines": [-20, -80]},
        "cci": {"zero_line": True},
        "macd": {"zero_line": True},
        "force_index": {"zero_line": True},
        "cmf": {"zero_line": True},
        "chaikin_oscillator": {"zero_line": True},
        "coppock": {"zero_line": True},
        "trix": {"zero_line": True},
        "tsi": {"zero_line": True},
        "klinger": {"zero_line": True},
        "fisher_transform": {"zero_line": True},
        "adx": {"y_range": [0, 100], "horizontal_lines": [25]},
        "aroon": {"y_range": [-100, 100], "zero_line": True},
        "vortex": {"horizontal_lines": [1.0]},
        "schaff_trend_cycle": {"y_range": [0, 100], "horizontal_lines": [25, 75]},
        "trend_intensity": {"y_range": [0, 100], "horizontal_lines": [50]},
    }

    def get_indicator_panels(
        self, datetimes: np.ndarray,
    ) -> dict:
        """Return indicator_panels structure for AlphaForge report charts.

        If subclass provides extended ``get_indicator_config()`` entries
        (with ``array`` field), panels are auto-generated.  Otherwise
        returns empty panels (subclasses can still override manually).
        """
        configs = self.get_indicator_config()
        if not configs:
            return {"overlays": [], "subplots": []}

        # Check if any config has "array" — if not, can't auto-generate
        has_arrays = any("array" in c for c in configs)
        if not has_arrays:
            return {"overlays": [], "subplots": []}

        overlays: list[dict] = []
        # Group subplot traces by panel name
        subplot_groups: dict[str, list] = {}
        subplot_meta: dict[str, dict] = {}

        for cfg in configs:
            arr = cfg.get("array")
            if arr is None:
                continue

            name = cfg.get("name", "indicator")
            style = cfg.get("style", "line")
            color = cfg.get("color")
            ind_type = cfg.get("type")

            # Auto-classify if type not specified
            if ind_type is None:
                # Extract base indicator name (lowercase, strip params)
                base = name.lower().split("(")[0].strip()
                ind_type = (
                    "overlay"
                    if any(base.startswith(kw) for kw in self._OVERLAY_INDICATORS)
                    else "subplot"
                )

            if ind_type == "overlay":
                overlays.append(self._make_overlay(
                    name, datetimes, arr, style=style, color=color,
                ))
            else:
                # Group by panel name (default: each indicator its own panel)
                panel_name = cfg.get("panel", name)

                if panel_name not in subplot_groups:
                    subplot_groups[panel_name] = []
                    # Collect panel-level metadata from config or hints
                    base = name.lower().split("(")[0].strip()
                    hints = {}
                    for hint_key, hint_val in self._SUBPLOT_HINTS.items():
                        if base.startswith(hint_key):
                            hints = hint_val
                            break
                    subplot_meta[panel_name] = {
                        "y_range": cfg.get("y_range", hints.get("y_range")),
                        "horizontal_lines": cfg.get("horizontal_lines",
                                                     hints.get("horizontal_lines")),
                        "zero_line": cfg.get("zero_line",
                                             hints.get("zero_line", False)),
                    }

                subplot_groups[panel_name].append(
                    self._make_subplot_trace(
                        name, datetimes, arr, style=style, color=color,
                        color_positive=cfg.get("color_positive"),
                        color_negative=cfg.get("color_negative"),
                    )
                )

        subplots = []
        for panel_name, traces in subplot_groups.items():
            meta = subplot_meta[panel_name]
            subplots.append(self._make_subplot(
                panel_name, traces, height_ratio=0.15,
                zero_line=meta.get("zero_line", False),
                horizontal_lines=meta.get("horizontal_lines"),
                y_range=meta.get("y_range"),
            ))

        return {"overlays": overlays, "subplots": subplots}

    # --- Helper methods for building panels ---

    @staticmethod
    def _make_overlay(
        name: str,
        datetimes: np.ndarray,
        data: np.ndarray,
        style: str = "line",
        color: str | None = None,
    ) -> dict:
        """Build an overlay trace dict for the indicator_panels spec.

        Args:
            name:      Display name (e.g. "EMA(20)").
            datetimes: Array of datetime values (shared x-axis).
            data:      Indicator values array (same length as datetimes).
            style:     "line", "step", or "dash".
            color:     Optional hex color (e.g. "#ffab40").
        """
        trace: dict = {
            "name": name,
            "data": list(zip(datetimes.tolist(), np.where(np.isnan(data), None, data).tolist())),
            "style": style,
        }
        if color is not None:
            trace["color"] = color
        return trace

    @staticmethod
    def _make_subplot(
        name: str,
        traces: list[dict],
        height_ratio: float = 0.15,
        zero_line: bool = False,
        horizontal_lines: list[float] | None = None,
        y_range: list[float] | None = None,
    ) -> dict:
        """Build a subplot panel dict for the indicator_panels spec.

        Args:
            name:             Panel title (e.g. "MACD").
            traces:           List of trace dicts (built via _make_subplot_trace).
            height_ratio:     Panel height as fraction of total chart (0.1-0.3).
            zero_line:        Draw a zero-axis line.
            horizontal_lines: List of y-values for reference lines (e.g. [30, 70]).
            y_range:          Fixed y-axis range (e.g. [0, 100]).
        """
        panel: dict = {
            "name": name,
            "height_ratio": height_ratio,
            "traces": traces,
        }
        if zero_line:
            panel["zero_line"] = True
        if horizontal_lines is not None:
            panel["horizontal_lines"] = horizontal_lines
        if y_range is not None:
            panel["y_range"] = y_range
        return panel

    @staticmethod
    def _make_subplot_trace(
        name: str,
        datetimes: np.ndarray,
        data: np.ndarray,
        style: str = "line",
        color: str | None = None,
        color_positive: str | None = None,
        color_negative: str | None = None,
    ) -> dict:
        """Build a single trace within a subplot panel.

        Args:
            name:           Trace name (e.g. "MACD Line").
            datetimes:      Datetime array.
            data:           Values array.
            style:          "line", "bar", "area", "step", or "dash".
            color:          Line/area color.
            color_positive: Bar color for positive values (bar style only).
            color_negative: Bar color for negative values (bar style only).
        """
        trace: dict = {
            "name": name,
            "data": list(zip(datetimes.tolist(), np.where(np.isnan(data), None, data).tolist())),
            "style": style,
        }
        if color is not None:
            trace["color"] = color
        if color_positive is not None:
            trace["color_positive"] = color_positive
        if color_negative is not None:
            trace["color_negative"] = color_negative
        return trace

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Enforce that concrete subclasses define required attributes."""
        super().__init_subclass__(**kwargs)
        # Skip enforcement on intermediate template classes
        if cls.__name__ in ("TrendingStrategy", "MeanReversionStrategy"):
            return
        required = ("name", "regime", "direction", "signal_dimensions", "warmup")
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
        if cls.direction not in cls._VALID_DIRECTIONS:
            raise TypeError(
                f"{cls.__name__}.direction must be 'long', 'short', or "
                f"'both', got '{cls.direction!r}'"
            )
        if cls.regime == "mean_reversion" and cls.direction != "both":
            raise TypeError(
                f"Mean reversion strategy {cls.__name__} must set "
                f"direction='both', got '{cls.direction!r}'"
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
