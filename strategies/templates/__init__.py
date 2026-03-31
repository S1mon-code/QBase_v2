"""Strategy templates for QBase_v2."""

from strategies.templates.base_strategy import QBaseStrategy
from strategies.templates.trending_template import TrendingStrategy
from strategies.templates.mean_reversion_template import MeanReversionStrategy

__all__ = ["QBaseStrategy", "TrendingStrategy", "MeanReversionStrategy"]
