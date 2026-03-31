"""Dynamic volatility classification based on ATR percentile rank.

high: >70th percentile
mid:  30th-70th percentile
low:  <30th percentile
"""

from __future__ import annotations


def classify_vol(atr_percentile: float) -> str:
    """Classify current volatility level.

    Parameters
    ----------
    atr_percentile : float
        ATR percentile rank in [0, 100].

    Returns
    -------
    str
        ``"high"`` if >70, ``"low"`` if <30, otherwise ``"mid"``.
    """
    if atr_percentile > 70.0:
        return "high"
    if atr_percentile < 30.0:
        return "low"
    return "mid"
