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
        The regime label set by fundamental analysis.
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
        Current regime label (e.g. ``"strong_trend"``, ``"mild_trend"``,
        ``"mean_reversion"``, ``"crisis"``).
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

    - Assigned ``mild_trend`` but vol > 90th percentile -> possible crisis.
    - Assigned ``mean_reversion`` but return > 15% -> possible trend.
    - Assigned ``strong_trend`` but ADX < 15 -> possibly ranging.
    - Assigned ``crisis`` but vol < 30th percentile and ADX < 20 -> possibly calm.
    """
    regime = assigned_regime.lower()

    # Mild trend but extremely high volatility -> could be crisis
    if regime in ("mild_trend", "mean_reversion") and recent_volatility_pctl > 90:
        return RegimeAlert(
            assigned_regime=assigned_regime,
            detected_behavior="extreme_volatility",
            severity="critical",
            message=(
                f"Assigned '{assigned_regime}' but volatility at {recent_volatility_pctl:.0f}th "
                f"percentile. Market behaviour suggests possible crisis regime."
            ),
        )

    # Mean reversion but large directional move
    if regime == "mean_reversion" and abs(recent_return_pct) > 15:
        return RegimeAlert(
            assigned_regime=assigned_regime,
            detected_behavior="strong_directional_move",
            severity="warning",
            message=(
                f"Assigned 'mean_reversion' but recent return is "
                f"{recent_return_pct:+.1f}%. Market may be trending."
            ),
        )

    # Strong trend but very weak trend strength
    if regime == "strong_trend" and recent_trend_strength < 15:
        return RegimeAlert(
            assigned_regime=assigned_regime,
            detected_behavior="weak_trend",
            severity="warning",
            message=(
                f"Assigned 'strong_trend' but ADX/trend strength is "
                f"{recent_trend_strength:.1f} (< 15). Market may be ranging."
            ),
        )

    # Mild trend but very strong trend
    if regime == "mild_trend" and recent_trend_strength > 40 and abs(recent_return_pct) > 10:
        return RegimeAlert(
            assigned_regime=assigned_regime,
            detected_behavior="possible_strong_trend",
            severity="info",
            message=(
                f"Assigned 'mild_trend' but trend strength is "
                f"{recent_trend_strength:.1f} with return {recent_return_pct:+.1f}%. "
                f"Consider upgrading to strong_trend."
            ),
        )

    # Crisis but calm market
    if regime == "crisis" and recent_volatility_pctl < 30 and recent_trend_strength < 20:
        return RegimeAlert(
            assigned_regime=assigned_regime,
            detected_behavior="calm_market",
            severity="warning",
            message=(
                f"Assigned 'crisis' but volatility at {recent_volatility_pctl:.0f}th "
                f"percentile and trend strength {recent_trend_strength:.1f}. "
                f"Market appears calm."
            ),
        )

    return None
