"""Template for trending strategies.

Trending strategies must include at least the Momentum signal dimension
and must set a horizon (fast / medium / slow).
"""

from __future__ import annotations

from strategies.templates.base_strategy import QBaseStrategy


class TrendingStrategy(QBaseStrategy):
    """Base template for all trending strategies.

    Sets ``regime = "trending"``.  Subclasses must set ``horizon`` to one
    of ``"fast"``, ``"medium"``, or ``"slow"`` and must include
    ``"momentum"`` in ``signal_dimensions``.
    """

    regime = "trending"
    # Subclass sets: horizon = "fast" | "medium" | "slow"
    horizon: str | None = None  # type: ignore[assignment]
