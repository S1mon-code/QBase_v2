"""Regime Mismatch Detection.

Compares the assigned fundamental regime with observed market behaviour
to detect mismatches that may require re-evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeAlert:
    """Alert indicating a mismatch between assigned regime and market behaviour.

    Attributes
    ----------
    assigned_regime : str
        The regime label set by fundamental analysis (``"long"`` or ``"short"``).
    detected_behavior : str
        Description of what the market is actually doing.
    severity : str
        ``"info"``, ``"warning"``, or ``"critical"``.
    message : str
        Human-readable explanation.
    """

    assigned_regime: str
    detected_behavior: str
    severity: str
    message: str


def check_regime_consistency(
    assigned_regime: str,
    recent_volatility_pctl: float,
    recent_trend_strength: float,
    recent_return_pct: float,
) -> RegimeAlert | None:
    """Check if market behaviour matches the assigned regime.

    Parameters
    ----------
    assigned_regime : str
        Current regime label (``"long"`` or ``"short"``).
    recent_volatility_pctl : float
        ATR percentile rank over lookback (0-100).
    recent_trend_strength : float
        Trend strength indicator value (e.g. ADX).
    recent_return_pct : float
        Recent-period return as percentage (e.g. 15.0 for 15%).

    Returns
    -------
    RegimeAlert | None
        Alert if mismatch detected, otherwise ``None``.

    Examples
    --------
    Mismatches detected:

    - Assigned ``long`` but large negative return -> market moving against regime.
    - Assigned ``short`` but large positive return -> market moving against regime.
    - Extreme volatility (> 90th percentile) -> warning regardless of regime.
    """
    regime = assigned_regime.lower()

    # Extreme volatility warning for any regime
    if recent_volatility_pctl > 90:
        return RegimeAlert(
            assigned_regime=assigned_regime,
            detected_behavior="extreme_volatility",
            severity="critical",
            message=(
                f"Assigned '{assigned_regime}' but volatility at {recent_volatility_pctl:.0f}th "
                f"percentile. Consider reducing position size."
            ),
        )

    # Long regime but market dropping significantly
    if regime == "long" and recent_return_pct < -10:
        return RegimeAlert(
            assigned_regime=assigned_regime,
            detected_behavior="against_regime",
            severity="warning",
            message=(
                f"Assigned 'long' but recent return is "
                f"{recent_return_pct:+.1f}%. Market moving against regime."
            ),
        )

    # Short regime but market rallying significantly
    if regime == "short" and recent_return_pct > 10:
        return RegimeAlert(
            assigned_regime=assigned_regime,
            detected_behavior="against_regime",
            severity="warning",
            message=(
                f"Assigned 'short' but recent return is "
                f"{recent_return_pct:+.1f}%. Market moving against regime."
            ),
        )

    # Weak trend strength warning
    if recent_trend_strength < 15:
        return RegimeAlert(
            assigned_regime=assigned_regime,
            detected_behavior="weak_trend",
            severity="info",
            message=(
                f"Assigned '{assigned_regime}' but trend strength is "
                f"{recent_trend_strength:.1f} (< 15). Market may be ranging."
            ),
        )

    return None
