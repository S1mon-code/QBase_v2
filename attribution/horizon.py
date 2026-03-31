"""Layer B: Horizon Attribution.

Regresses strategy returns on TSMOM factors (1M, 3M, 12M) to determine
which time horizon drives performance and how much is independent alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HorizonAttributionResult:
    """Result of horizon attribution regression."""

    beta_fast: float  # loading on TSMOM 1M
    beta_medium: float  # loading on TSMOM 3M
    beta_slow: float  # loading on TSMOM 12M
    independent_alpha: float  # intercept (annualized)
    r_squared: float
    horizon_fingerprint: dict[str, float]  # {"fast": pct, "medium": pct, "slow": pct}


def horizon_attribution(
    strategy_returns: np.ndarray,
    tsmom_1m_returns: np.ndarray,
    tsmom_3m_returns: np.ndarray,
    tsmom_12m_returns: np.ndarray,
    annualize: int = 252,
) -> HorizonAttributionResult:
    """Regress strategy returns on TSMOM factors via OLS.

    strategy_returns = alpha + beta_fast*TSMOM_1M + beta_medium*TSMOM_3M
                       + beta_slow*TSMOM_12M + epsilon

    Args:
        strategy_returns: 1-D array of strategy daily returns.
        tsmom_1m_returns: 1-D array of TSMOM 1-month factor returns.
        tsmom_3m_returns: 1-D array of TSMOM 3-month factor returns.
        tsmom_12m_returns: 1-D array of TSMOM 12-month factor returns.
        annualize: Trading days per year for annualizing alpha.

    Returns:
        HorizonAttributionResult with betas, alpha, R-squared, and fingerprint.

    Raises:
        ValueError: If array lengths do not match or are empty.
    """
    y = np.asarray(strategy_returns, dtype=np.float64).ravel()
    x1 = np.asarray(tsmom_1m_returns, dtype=np.float64).ravel()
    x2 = np.asarray(tsmom_3m_returns, dtype=np.float64).ravel()
    x3 = np.asarray(tsmom_12m_returns, dtype=np.float64).ravel()

    n = len(y)
    if n == 0:
        raise ValueError("strategy_returns must not be empty")
    if not (len(x1) == len(x2) == len(x3) == n):
        raise ValueError("All return arrays must have the same length")

    # Build design matrix with intercept
    X = np.column_stack([np.ones(n), x1, x2, x3])

    # OLS via numpy lstsq
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily = coeffs[0]
    beta_fast = coeffs[1]
    beta_medium = coeffs[2]
    beta_slow = coeffs[3]

    # Annualize alpha
    independent_alpha = alpha_daily * annualize

    # Compute R-squared
    y_hat = X @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    # Horizon fingerprint: proportion based on absolute betas
    abs_betas = np.array([abs(beta_fast), abs(beta_medium), abs(beta_slow)])
    total_abs = abs_betas.sum()
    if total_abs > 0.0:
        pcts = abs_betas / total_abs * 100.0
    else:
        pcts = np.array([0.0, 0.0, 0.0])

    horizon_fingerprint = {
        "fast": float(pcts[0]),
        "medium": float(pcts[1]),
        "slow": float(pcts[2]),
    }

    return HorizonAttributionResult(
        beta_fast=float(beta_fast),
        beta_medium=float(beta_medium),
        beta_slow=float(beta_slow),
        independent_alpha=float(independent_alpha),
        r_squared=float(r_squared),
        horizon_fingerprint=horizon_fingerprint,
    )
