"""5-dimension composite objective function for strategy optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optimizer.config import MIN_TRADES, WEIGHTS


@dataclass(frozen=True)
class BacktestMetrics:
    """Metrics from a single backtest run."""

    sharpe: float
    max_drawdown: float  # positive fraction (0.15 = 15%)
    cvar_95: float  # negative (worst 5% mean daily return)
    n_trades: int
    win_rate: float  # monthly positive ratio
    skewness: float
    kurtosis: float
    daily_returns: np.ndarray  # for consistency analysis
    equity_curve: np.ndarray  # for quality analysis


def _score_performance(sharpe: float, phase: str) -> float:
    """S_performance: risk-adjusted return score (0-10).

    Coarse phase: tanh compression to avoid chasing extreme Sharpe.
    Fine phase: linear scaling for absolute precision.
    """
    if phase == "coarse":
        return 10.0 * math.tanh(0.7 * sharpe)
    # fine
    return min(10.0, sharpe * 10.0 / 3.0)


def _score_significance(sharpe: float, n_trades: int, skewness: float, kurtosis: float) -> float:
    """S_significance: statistical significance via Lo (2002) adjusted t-stat (0-10).

    t_stat = sharpe * sqrt(n_trades) / sqrt(1 + 0.5*skew*sharpe - (kurt-3)/4*sharpe^2)
    Score = min(10, t_stat * 10/3)  → t > 3 = full score.
    """
    if n_trades <= 0:
        return 0.0

    denominator_inner = 1.0 + 0.5 * skewness * sharpe - (kurtosis - 3.0) / 4.0 * sharpe ** 2
    # Guard against negative denominator (extreme skew/kurtosis)
    if denominator_inner <= 0:
        denominator_inner = 1.0

    t_stat = sharpe * math.sqrt(n_trades) / math.sqrt(denominator_inner)
    return max(0.0, min(10.0, t_stat * 10.0 / 3.0))


def _score_consistency(daily_returns: np.ndarray, n_windows: int = 5) -> float:
    """S_consistency: time consistency score (0-10).

    Split returns into N equal windows, compute per-window Sharpe.
    win_rate = fraction of windows with Sharpe > 0
    cv = std(window_sharpes) / mean(window_sharpes)
    consistency = win_rate * max(0, 1 - cv)
    """
    if len(daily_returns) < n_windows:
        return 0.0

    windows = np.array_split(daily_returns, n_windows)
    window_sharpes = []
    for w in windows:
        if len(w) == 0:
            window_sharpes.append(0.0)
        elif np.std(w) == 0:
            # Constant returns: positive mean → treat as positive Sharpe
            window_sharpes.append(10.0 if np.mean(w) > 0 else 0.0)
        else:
            window_sharpes.append(float(np.mean(w) / np.std(w)))

    sharpe_arr = np.array(window_sharpes)
    win_rate = float(np.mean(sharpe_arr > 0))
    mean_sharpe = float(np.mean(sharpe_arr))

    if mean_sharpe == 0:
        return 0.0

    cv = float(np.std(sharpe_arr) / abs(mean_sharpe))
    consistency = win_rate * max(0.0, 1.0 - cv)
    return 10.0 * consistency


def _score_risk(max_drawdown: float, cvar_95: float) -> float:
    """S_risk: tail risk score (0-10).

    maxdd_score = max(0, 10 * (1 - |maxdd| / 0.40))  → 40% DD = 0
    cvar_score  = max(0, 10 * (1 + cvar_95 / 0.03))   → CVaR -3% = 0
    S_risk = 0.6 * maxdd_score + 0.4 * cvar_score
    """
    maxdd_score = max(0.0, 10.0 * (1.0 - abs(max_drawdown) / 0.40))
    cvar_score = max(0.0, 10.0 * (1.0 + cvar_95 / 0.03))
    return 0.6 * maxdd_score + 0.4 * cvar_score


def _score_alpha(strategy_sharpe: float, baseline_sharpe: float) -> float:
    """S_alpha: excess return over baseline (0-10).

    alpha_sharpe = strategy_sharpe - baseline_sharpe
    S_alpha = max(0, min(10, alpha_sharpe * 10 / 1.0))  → alpha > 1.0 = full score
    """
    alpha_sharpe = strategy_sharpe - baseline_sharpe
    return max(0.0, min(10.0, alpha_sharpe * 10.0 / 1.0))


def composite_objective(
    metrics: BacktestMetrics,
    baseline_sharpe: float = 0.0,
    phase: str = "coarse",
    freq: str = "1h",
) -> float:
    """5-dimension composite score (0-10 scale).

    score = 0.40 * S_performance + 0.15 * S_significance
          + 0.15 * S_consistency + 0.15 * S_risk + 0.15 * S_alpha

    Returns -10.0 for hard-filtered strategies (insufficient trades).
    Returns -5.0 if alpha <= 0 (worse than baseline).
    """
    # Hard filter: minimum trade count
    min_trades = MIN_TRADES.get(freq, 30)
    if metrics.n_trades < min_trades:
        return -10.0

    # Hard filter: alpha must be positive
    alpha_sharpe = metrics.sharpe - baseline_sharpe
    if alpha_sharpe <= 0:
        return -5.0

    s_perf = _score_performance(metrics.sharpe, phase)
    s_sig = _score_significance(
        metrics.sharpe, metrics.n_trades, metrics.skewness, metrics.kurtosis,
    )
    s_con = _score_consistency(metrics.daily_returns)
    s_risk = _score_risk(metrics.max_drawdown, metrics.cvar_95)
    s_alpha = _score_alpha(metrics.sharpe, baseline_sharpe)

    score = (
        WEIGHTS["performance"] * s_perf
        + WEIGHTS["significance"] * s_sig
        + WEIGHTS["consistency"] * s_con
        + WEIGHTS["risk"] * s_risk
        + WEIGHTS["alpha"] * s_alpha
    )
    return score
