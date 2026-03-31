"""Template for mean-reversion strategies.

Mean-reversion strategies typically rely on Technical or Volume/OI dimensions
and do not use a trend horizon.
"""

from __future__ import annotations

from strategies.templates.base_strategy import QBaseStrategy


class MeanReversionStrategy(QBaseStrategy):
    """Base template for all mean-reversion strategies.

    Sets ``regime = "mean_reversion"`` and ``horizon = None``.
    """

    regime = "mean_reversion"
    horizon = None
