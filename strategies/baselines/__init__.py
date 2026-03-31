"""TSMOM Baseline strategies."""

from strategies.baselines.tsmom_fast import TSMOMFast
from strategies.baselines.tsmom_medium import TSMOMMedium
from strategies.baselines.tsmom_slow import TSMOMSlow

__all__ = ["TSMOMFast", "TSMOMMedium", "TSMOMSlow"]
