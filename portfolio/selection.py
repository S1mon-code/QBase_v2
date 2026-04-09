"""Strategy Selection Filters.

Two-stage filtering for portfolio inclusion:
    Stage 1: Hard filters (pass/fail) — any failure = rejection
    Stage 2: Portfolio fit (correlation + marginal Sharpe) — determines admission

Trade count → confidence grading (affects weight cap, not admission):
    HIGH_CONFIDENCE:  trades >= 30 (daily) or >= 100 (1h)
    MODERATE:         trades >= 10
    LOW_CONFIDENCE:   trades < 10 → weight capped at 15%
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Confidence(str, Enum):
    """Trade-count confidence grading."""

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


# Max weight per confidence level
CONFIDENCE_WEIGHT_CAP: dict[Confidence, float] = {
    Confidence.HIGH: 0.25,
    Confidence.MODERATE: 0.25,
    Confidence.LOW: 0.15,
}

# Trade count thresholds by frequency
CONFIDENCE_THRESHOLDS: dict[str, dict[str, int]] = {
    "daily": {"high": 30, "moderate": 10},
    "4h": {"high": 50, "moderate": 20},
    "2h": {"high": 60, "moderate": 25},
    "1h": {"high": 100, "moderate": 30},
}


@dataclass(frozen=True)
class SelectionResult:
    """Result of strategy selection for a single strategy.

    Attributes
    ----------
    strategy : str
        Strategy name.
    passed : bool
        Whether the strategy passed all hard filters.
    confidence : Confidence
        Trade-count confidence grading.
    max_weight : float
        Maximum allowed weight based on confidence.
    rejection_reasons : tuple[str, ...]
        Reasons for rejection (empty if passed).
    warnings : tuple[str, ...]
        Non-blocking warnings.
    """

    strategy: str
    passed: bool
    confidence: Confidence
    max_weight: float
    rejection_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _grade_confidence(n_trades: int, freq: str = "daily") -> Confidence:
    """Grade confidence based on trade count and frequency.

    Parameters
    ----------
    n_trades : int
        Number of trades in OOS period.
    freq : str
        Strategy frequency: ``"daily"``, ``"4h"``, ``"2h"``, ``"1h"``.

    Returns
    -------
    Confidence
        Confidence grading.
    """
    thresholds = CONFIDENCE_THRESHOLDS.get(freq, CONFIDENCE_THRESHOLDS["daily"])
    if n_trades >= thresholds["high"]:
        return Confidence.HIGH
    if n_trades >= thresholds["moderate"]:
        return Confidence.MODERATE
    return Confidence.LOW


def select_strategies(
    candidates: dict[str, dict],
    min_activity: float = 0.001,
    require_dsr: float = 0.95,
    min_oos_sharpe: float = 0.5,
    max_drawdown: float = -0.25,
    min_industrial_sharpe: float = 0.5,
    max_industrial_decay: float = 0.50,
    min_profit_factor: float = 1.3,
    max_single_trade_pct: float = 0.30,
) -> dict[str, SelectionResult]:
    """Filter strategies for portfolio inclusion.

    Hard filters (ALL must pass):
        1. OOS annualized return > 0 (must make money)
        2. OOS Sharpe (industrial) >= *min_oos_sharpe*
        3. Industrial Sharpe >= *min_industrial_sharpe*
        4. Industrial decay <= *max_industrial_decay*
        5. Max drawdown >= *max_drawdown* (less negative)
        6. DSR >= *require_dsr*
        7. Bootstrap verdict != ``"FRAGILE"``
        8. Regime CV verdict != ``"FAIL"``
        9. Independent alpha > 0
       10. Activity > *min_activity*
       11. 2x cost survival: strategy profitable at 2x estimated costs
       12. Profit Factor >= *min_profit_factor*

    Parameters
    ----------
    candidates : dict[str, dict]
        Mapping of strategy name to info dict. Expected keys:

        - ``"validation"`` : object with attrs ``regime_cv``, ``deflated_sharpe``,
          ``bootstrap``, ``industrial``, ``oos_sharpe``, ``max_drawdown``,
          ``annualized_return``, ``profit_factor``, ``max_single_trade_pct``
        - ``"alpha"`` : float (independent alpha)
        - ``"activity"`` : float (average absolute daily return)
        - ``"n_trades"`` : int (number of trades in OOS)
        - ``"freq"`` : str (``"daily"``, ``"1h"``, etc.)
        - ``"sharpe_at_2x_cost"`` : float (Sharpe at 2x transaction costs)

    min_activity : float
        Minimum activity threshold.
    require_dsr : float
        Minimum deflated Sharpe ratio probability.
    min_oos_sharpe : float
        Minimum OOS Sharpe ratio (post-cost).
    max_drawdown : float
        Maximum allowed drawdown (negative, e.g. -0.25).
    min_industrial_sharpe : float
        Minimum industrial Sharpe ratio.
    max_industrial_decay : float
        Maximum allowed industrial decay (0-1).

    Returns
    -------
    dict[str, SelectionResult]
        Mapping of strategy name to selection result (both passed and failed).
    """
    results: dict[str, SelectionResult] = {}

    for name, info in candidates.items():
        validation = info.get("validation")
        alpha = info.get("alpha", 0.0)
        activity = info.get("activity", 0.0)
        n_trades = info.get("n_trades", 0)
        freq = info.get("freq", "daily")
        sharpe_2x = info.get("sharpe_at_2x_cost")

        rejections: list[str] = []
        warnings: list[str] = []

        # Confidence grading
        confidence = _grade_confidence(n_trades, freq)
        weight_cap = CONFIDENCE_WEIGHT_CAP[confidence]

        if confidence == Confidence.LOW:
            warnings.append(
                f"LOW confidence: only {n_trades} trades (freq={freq}), weight capped at 15%"
            )

        # --- Hard filters ---

        # Activity filter
        if activity <= min_activity:
            rejections.append(f"Activity {activity:.4f} <= {min_activity}")

        # Alpha filter
        if alpha <= 0:
            rejections.append(f"Independent alpha {alpha:.4f} <= 0")

        if validation is None:
            rejections.append("No validation data")
            results[name] = SelectionResult(
                strategy=name,
                passed=False,
                confidence=confidence,
                max_weight=weight_cap,
                rejection_reasons=tuple(rejections),
                warnings=tuple(warnings),
            )
            continue

        # Absolute return filter (must make money)
        ann_return = getattr(validation, "annualized_return", None)
        if ann_return is not None and ann_return <= 0:
            rejections.append(f"OOS annualized return {ann_return:.2%} <= 0 (negative return)")

        # OOS Sharpe filter
        oos_sharpe = getattr(validation, "oos_sharpe", None)
        if oos_sharpe is not None and oos_sharpe < min_oos_sharpe:
            rejections.append(f"OOS Sharpe {oos_sharpe:.3f} < {min_oos_sharpe}")

        # High Sharpe but low return warning
        if (oos_sharpe is not None and ann_return is not None
                and oos_sharpe > 1.5 and ann_return < 0.03):
            warnings.append(
                f"High Sharpe ({oos_sharpe:.2f}) but low return ({ann_return:.1%}) — "
                f"may be low-activity strategy"
            )

        # Profit Factor filter
        pf = getattr(validation, "profit_factor", None)
        if pf is not None and pf < min_profit_factor:
            rejections.append(f"Profit Factor {pf:.2f} < {min_profit_factor}")

        # Single trade concentration filter
        single_trade = getattr(validation, "max_single_trade_pct", None)
        if single_trade is not None and single_trade > max_single_trade_pct:
            rejections.append(
                f"Top single trade {single_trade:.1%} > {max_single_trade_pct:.0%} of total P&L"
            )

        # Max drawdown filter
        oos_max_dd = getattr(validation, "max_drawdown", None)
        if oos_max_dd is not None and oos_max_dd < max_drawdown:
            rejections.append(f"Max DD {oos_max_dd:.2%} worse than {max_drawdown:.0%}")

        # Regime CV filter
        regime_cv = getattr(validation, "regime_cv", None)
        if regime_cv is not None and getattr(regime_cv, "verdict", "") == "FAIL":
            rejections.append("Regime CV = FAIL")

        # Industrial Sharpe filter
        industrial = getattr(validation, "industrial", None)
        if industrial is not None:
            ind_sharpe = getattr(industrial, "industrial_sharpe", None)
            if ind_sharpe is not None and ind_sharpe < min_industrial_sharpe:
                rejections.append(
                    f"Industrial Sharpe {ind_sharpe:.3f} < {min_industrial_sharpe}"
                )

            # Industrial decay filter
            decay_pct = getattr(industrial, "decay_pct", None)
            if decay_pct is not None and decay_pct > max_industrial_decay:
                rejections.append(
                    f"Industrial decay {decay_pct:.1%} > {max_industrial_decay:.0%}"
                )

        # Deflated Sharpe filter
        dsr = getattr(validation, "deflated_sharpe", None)
        if dsr is not None and dsr < require_dsr:
            rejections.append(f"DSR {dsr:.4f} < {require_dsr}")

        # Bootstrap filter
        bootstrap = getattr(validation, "bootstrap", None)
        if bootstrap is not None and getattr(bootstrap, "verdict", "") == "FRAGILE":
            rejections.append("Bootstrap = FRAGILE")

        # 2x cost survival filter
        if sharpe_2x is not None and sharpe_2x <= 0:
            rejections.append(f"2x cost Sharpe {sharpe_2x:.3f} <= 0 (not profitable at 2x costs)")

        results[name] = SelectionResult(
            strategy=name,
            passed=len(rejections) == 0,
            confidence=confidence,
            max_weight=weight_cap,
            rejection_reasons=tuple(rejections),
            warnings=tuple(warnings),
        )

    return results


def get_passed_strategies(results: dict[str, SelectionResult]) -> list[str]:
    """Extract sorted list of strategy names that passed all hard filters.

    Parameters
    ----------
    results : dict[str, SelectionResult]
        Output from ``select_strategies``.

    Returns
    -------
    list[str]
        Sorted strategy names that passed.
    """
    return sorted(name for name, r in results.items() if r.passed)


def check_portfolio_fit(
    candidate_returns: np.ndarray,
    portfolio_returns: np.ndarray,
    candidate_sharpe: float,
    portfolio_sharpe: float,
    max_correlation: float = 0.40,
) -> tuple[bool, dict[str, float]]:
    """Check if a candidate strategy fits the existing portfolio.

    Two checks:
        1. Correlation with existing portfolio < *max_correlation*
        2. Marginal Sharpe contribution > 0
           (candidate SR > correlation * portfolio SR)

    Parameters
    ----------
    candidate_returns : np.ndarray
        Daily returns of the candidate strategy, shape (n_days,).
    portfolio_returns : np.ndarray
        Daily returns of the existing portfolio, shape (n_days,).
    candidate_sharpe : float
        Annualized Sharpe ratio of the candidate.
    portfolio_sharpe : float
        Annualized Sharpe ratio of the existing portfolio.
    max_correlation : float
        Maximum allowed correlation.

    Returns
    -------
    tuple[bool, dict[str, float]]
        (fits, details) where details contains ``"correlation"``,
        ``"marginal_sharpe_threshold"``, ``"candidate_sharpe"``,
        and ``"fits"``.
    """
    # Align lengths
    n = min(len(candidate_returns), len(portfolio_returns))
    if n < 20:
        # Not enough data to compute meaningful correlation
        return True, {
            "correlation": 0.0,
            "marginal_sharpe_threshold": 0.0,
            "candidate_sharpe": candidate_sharpe,
            "fits": True,
            "reason": "insufficient data for correlation, admitted by default",
        }

    c_ret = candidate_returns[-n:]
    p_ret = portfolio_returns[-n:]

    # Compute correlation
    corr = float(np.corrcoef(c_ret, p_ret)[0, 1])
    if np.isnan(corr):
        corr = 0.0

    # Marginal Sharpe threshold: candidate needs SR > rho * portfolio_SR
    marginal_threshold = corr * portfolio_sharpe

    # Check both conditions
    corr_ok = abs(corr) < max_correlation
    marginal_ok = candidate_sharpe > marginal_threshold

    fits = corr_ok and marginal_ok

    details = {
        "correlation": corr,
        "marginal_sharpe_threshold": marginal_threshold,
        "candidate_sharpe": candidate_sharpe,
        "fits": fits,
    }

    if not corr_ok:
        details["reason"] = f"Correlation {corr:.3f} >= {max_correlation}"
    elif not marginal_ok:
        details["reason"] = (
            f"Candidate SR {candidate_sharpe:.3f} <= "
            f"rho*portfolio_SR {marginal_threshold:.3f}"
        )
    else:
        details["reason"] = "passed"

    return fits, details


def check_pairwise_correlations(
    returns_matrix: np.ndarray,
    strategy_names: list[str],
    max_correlation: float = 0.40,
) -> dict[str, list[str]]:
    """Check pairwise correlations and flag highly correlated pairs.

    Parameters
    ----------
    returns_matrix : np.ndarray
        Returns matrix of shape (n_periods, n_strategies).
    strategy_names : list[str]
        Strategy names corresponding to columns.
    max_correlation : float
        Threshold for flagging.

    Returns
    -------
    dict[str, list[str]]
        Mapping of strategy name to list of strategies it is highly
        correlated with (correlation >= *max_correlation*).
    """
    n = len(strategy_names)
    if n < 2:
        return {s: [] for s in strategy_names}

    corr_matrix = np.corrcoef(returns_matrix.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    flags: dict[str, list[str]] = {s: [] for s in strategy_names}

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) >= max_correlation:
                flags[strategy_names[i]].append(strategy_names[j])
                flags[strategy_names[j]].append(strategy_names[i])

    return flags
