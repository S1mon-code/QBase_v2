"""5-Dimension 15-Metric Portfolio Scoring.

Dimensions and weights:
    - Return/Risk:      35%
    - Signal Quality:   25%
    - Efficiency:       20%
    - Robustness:       15%
    - Operability:       5%

Grade mapping: 90+ -> A+, 85+ -> A, 80+ -> A-, 75+ -> B+,
               70+ -> B, 60+ -> C, < 60 -> D/F

Passing threshold: >= 75 (B+)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortfolioScore:
    """Portfolio scoring result.

    Attributes
    ----------
    total : float
        Overall score 0-100.
    grade : str
        Letter grade from A+ to D/F.
    dimensions : dict[str, float]
        Per-dimension score (0-100).
    metrics : dict[str, float]
        Raw metric values.
    passed : bool
        Whether total >= 75 (B+ threshold).
    """

    total: float
    grade: str
    dimensions: dict[str, float]
    metrics: dict[str, float]
    passed: bool


def _grade_from_score(score: float) -> str:
    """Map numeric score to letter grade.

    Parameters
    ----------
    score : float
        Score in [0, 100].

    Returns
    -------
    str
        Letter grade.
    """
    if score >= 90:
        return "A+"
    if score >= 85:
        return "A"
    if score >= 80:
        return "A-"
    if score >= 75:
        return "B+"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    return "D/F"


def _score_metric(value: float, thresholds: list[tuple[float, float]]) -> float:
    """Score a single metric using linear interpolation between thresholds.

    Parameters
    ----------
    value : float
        Raw metric value.
    thresholds : list[tuple[float, float]]
        List of (metric_value, score) pairs sorted by metric_value ascending.
        Score is linearly interpolated between adjacent pairs.

    Returns
    -------
    float
        Score in [0, 100].
    """
    if not thresholds:
        return 50.0

    if value <= thresholds[0][0]:
        return thresholds[0][1]
    if value >= thresholds[-1][0]:
        return thresholds[-1][1]

    for i in range(len(thresholds) - 1):
        v0, s0 = thresholds[i]
        v1, s1 = thresholds[i + 1]
        if v0 <= value <= v1:
            if v1 == v0:
                return s0
            t = (value - v0) / (v1 - v0)
            return s0 + t * (s1 - s0)

    return thresholds[-1][1]


def score_portfolio(
    sharpe: float,
    calmar: float,
    max_dd: float,
    dd_duration: int,
    cvar_95: float,
    avg_indep_alpha: float,
    horizon_diversity: float,
    vs_tsmom_excess: float,
    avg_correlation: float,
    dd_overlap: float,
    portfolio_vs_best: float,
    pos_sharpe_pct: float,
    bootstrap_ci_width: float,
    core_pct: float,
    permutation_p: float,
    n_strategies: int,
    max_weight: float,
    industrial_decay: float,
) -> PortfolioScore:
    """Score portfolio on 5 dimensions, 15 metrics.

    Parameters
    ----------
    sharpe : float
        Portfolio Sharpe ratio.
    calmar : float
        Portfolio Calmar ratio.
    max_dd : float
        Maximum drawdown (negative, e.g. -0.15).
    dd_duration : int
        Longest drawdown duration in days.
    cvar_95 : float
        95% Conditional Value at Risk (negative).
    avg_indep_alpha : float
        Average independent alpha across strategies.
    horizon_diversity : float
        Horizon diversity score (0-1, 1 = perfectly diversified).
    vs_tsmom_excess : float
        Excess return vs TSMOM benchmark.
    avg_correlation : float
        Average pairwise strategy correlation.
    dd_overlap : float
        Drawdown overlap ratio (0-1, lower is better).
    portfolio_vs_best : float
        Portfolio Sharpe / best single strategy Sharpe.
    pos_sharpe_pct : float
        Percentage of strategies with positive Sharpe (0-1).
    bootstrap_ci_width : float
        Bootstrap confidence interval width.
    core_pct : float
        Percentage of CORE strategies in stability test (0-1).
    permutation_p : float
        Permutation test p-value.
    n_strategies : int
        Number of strategies in portfolio.
    max_weight : float
        Maximum single strategy weight.
    industrial_decay : float
        Average industrial decay (0-1, lower is better).

    Returns
    -------
    PortfolioScore
        Scored result with grade.
    """
    metrics = {
        "sharpe": sharpe,
        "calmar": calmar,
        "max_dd": max_dd,
        "dd_duration": dd_duration,
        "cvar_95": cvar_95,
        "avg_indep_alpha": avg_indep_alpha,
        "horizon_diversity": horizon_diversity,
        "vs_tsmom_excess": vs_tsmom_excess,
        "avg_correlation": avg_correlation,
        "dd_overlap": dd_overlap,
        "portfolio_vs_best": portfolio_vs_best,
        "pos_sharpe_pct": pos_sharpe_pct,
        "bootstrap_ci_width": bootstrap_ci_width,
        "core_pct": core_pct,
        "permutation_p": permutation_p,
        "n_strategies": n_strategies,
        "max_weight": max_weight,
        "industrial_decay": industrial_decay,
    }

    # --- Return/Risk (35%) ---
    s_sharpe = _score_metric(sharpe, [(0.0, 0), (0.5, 40), (1.0, 70), (1.5, 85), (2.0, 100)])
    s_calmar = _score_metric(calmar, [(0.0, 0), (0.5, 40), (1.0, 70), (2.0, 90), (3.0, 100)])
    s_maxdd = _score_metric(-max_dd, [(0.0, 100), (0.1, 80), (0.2, 60), (0.3, 30), (0.5, 0)])
    s_dd_dur = _score_metric(float(dd_duration), [(0, 100), (30, 80), (60, 60), (120, 30), (250, 0)])
    s_cvar = _score_metric(-cvar_95, [(0.0, 100), (0.02, 80), (0.04, 60), (0.08, 30), (0.15, 0)])
    return_risk = (s_sharpe + s_calmar + s_maxdd + s_dd_dur + s_cvar) / 5.0

    # --- Signal Quality (25%) ---
    s_alpha = _score_metric(avg_indep_alpha, [(0.0, 0), (0.01, 50), (0.03, 75), (0.05, 90), (0.1, 100)])
    s_horizon = _score_metric(horizon_diversity, [(0.0, 0), (0.3, 40), (0.6, 70), (0.8, 85), (1.0, 100)])
    s_tsmom = _score_metric(vs_tsmom_excess, [(0.0, 30), (0.01, 50), (0.03, 70), (0.05, 85), (0.1, 100)])
    signal_quality = (s_alpha + s_horizon + s_tsmom) / 3.0

    # --- Efficiency (20%) ---
    s_corr = _score_metric(avg_correlation, [(0.0, 100), (0.2, 80), (0.4, 60), (0.6, 30), (0.8, 0)])
    s_overlap = _score_metric(dd_overlap, [(0.0, 100), (0.2, 80), (0.4, 60), (0.6, 30), (0.8, 0)])
    s_pvb = _score_metric(portfolio_vs_best, [(0.5, 0), (0.8, 40), (1.0, 70), (1.2, 85), (1.5, 100)])
    s_pos = _score_metric(pos_sharpe_pct, [(0.0, 0), (0.5, 40), (0.7, 60), (0.8, 80), (1.0, 100)])
    efficiency = (s_corr + s_overlap + s_pvb + s_pos) / 4.0

    # --- Robustness (15%) ---
    s_boot = _score_metric(bootstrap_ci_width, [(0.0, 100), (0.3, 80), (0.5, 60), (1.0, 30), (2.0, 0)])
    s_core = _score_metric(core_pct, [(0.0, 0), (0.3, 30), (0.5, 60), (0.7, 80), (1.0, 100)])
    s_perm = _score_metric(permutation_p, [(0.0, 100), (0.01, 85), (0.05, 60), (0.1, 30), (0.2, 0)])
    robustness = (s_boot + s_core + s_perm) / 3.0

    # --- Operability (5%) ---
    s_nstrat = _score_metric(float(n_strategies), [(1, 20), (3, 60), (5, 80), (8, 90), (12, 100)])
    s_maxw = _score_metric(max_weight, [(0.1, 100), (0.15, 90), (0.2, 80), (0.25, 70), (0.5, 30), (1.0, 0)])
    s_decay = _score_metric(industrial_decay, [(0.0, 100), (0.1, 85), (0.2, 70), (0.3, 50), (0.5, 0)])
    operability = (s_nstrat + s_maxw + s_decay) / 3.0

    dimensions = {
        "return_risk": return_risk,
        "signal_quality": signal_quality,
        "efficiency": efficiency,
        "robustness": robustness,
        "operability": operability,
    }

    total = (
        return_risk * 0.35
        + signal_quality * 0.25
        + efficiency * 0.20
        + robustness * 0.15
        + operability * 0.05
    )

    total = float(np.clip(total, 0.0, 100.0))
    grade = _grade_from_score(total)

    return PortfolioScore(
        total=total,
        grade=grade,
        dimensions=dimensions,
        metrics=metrics,
        passed=total >= 75.0,
    )
