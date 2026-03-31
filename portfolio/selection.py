"""Strategy Selection Filters.

Filters strategies for portfolio inclusion based on validation results,
alpha, and activity thresholds.
"""

from __future__ import annotations


def select_strategies(
    candidates: dict[str, dict],
    min_activity: float = 0.001,
    require_dsr: float = 0.95,
) -> list[str]:
    """Filter strategies for portfolio inclusion.

    Hard filters applied:
        - ``regime_cv`` verdict is not ``"FAIL"``
        - Industrial Sharpe > 0
        - Deflated Sharpe Ratio > *require_dsr*
        - Bootstrap verdict is not ``"FRAGILE"``
        - Activity (abs daily return) > *min_activity*
        - Independent alpha > 0

    Parameters
    ----------
    candidates : dict[str, dict]
        Mapping of strategy name to info dict. Expected keys:

        - ``"validation"`` : object with attrs ``regime_cv``, ``deflated_sharpe``,
          ``bootstrap``, ``industrial``
        - ``"alpha"`` : float (independent alpha)
        - ``"activity"`` : float (average absolute daily return)

    min_activity : float
        Minimum activity threshold.
    require_dsr : float
        Minimum deflated Sharpe ratio probability.

    Returns
    -------
    list[str]
        Sorted list of strategy names that pass all hard filters.
    """
    passed: list[str] = []

    for name, info in candidates.items():
        validation = info.get("validation")
        alpha = info.get("alpha", 0.0)
        activity = info.get("activity", 0.0)

        # Activity filter
        if activity <= min_activity:
            continue

        # Alpha filter
        if alpha <= 0:
            continue

        if validation is None:
            continue

        # Regime CV filter
        regime_cv = getattr(validation, "regime_cv", None)
        if regime_cv is not None and getattr(regime_cv, "verdict", "") == "FAIL":
            continue

        # Industrial Sharpe filter
        industrial = getattr(validation, "industrial", None)
        if industrial is not None:
            ind_sharpe = getattr(industrial, "industrial_sharpe", None)
            if ind_sharpe is not None and ind_sharpe <= 0:
                continue

        # Deflated Sharpe filter
        dsr = getattr(validation, "deflated_sharpe", None)
        if dsr is not None and dsr < require_dsr:
            continue

        # Bootstrap filter
        bootstrap = getattr(validation, "bootstrap", None)
        if bootstrap is not None and getattr(bootstrap, "verdict", "") == "FRAGILE":
            continue

        passed.append(name)

    return sorted(passed)
