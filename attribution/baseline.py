"""Layer D: Baseline Decomposition.

Decomposes strategy returns into TSMOM beta, Carry beta, and independent alpha
to determine how much of the return is genuine alpha vs. replicable beta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaselineDecomposition:
    """Result of baseline factor decomposition."""

    total_return_ann: float
    tsmom_beta_return: float  # beta_tsmom * TSMOM return
    carry_beta_return: float  # beta_carry * Carry return
    independent_alpha: float  # intercept (annualized)
    tsmom_pct: float  # % of total from TSMOM
    carry_pct: float
    alpha_pct: float
    r_squared: float


def decompose_baseline(
    strategy_returns: np.ndarray,
    tsmom_returns: np.ndarray,
    carry_returns: np.ndarray | None = None,
    annualize: int = 252,
) -> BaselineDecomposition:
    """Regress strategy returns on TSMOM + Carry baselines.

    strategy_return = alpha + beta_tsmom*TSMOM + beta_carry*Carry + epsilon

    If carry_returns is None, only TSMOM is used.

    Args:
        strategy_returns: 1-D array of strategy daily returns.
        tsmom_returns: 1-D array of TSMOM benchmark returns.
        carry_returns: Optional 1-D array of Carry benchmark returns.
        annualize: Trading days per year.

    Returns:
        BaselineDecomposition with factor contributions and percentages.

    Raises:
        ValueError: If arrays are empty or have mismatched lengths.
    """
    y = np.asarray(strategy_returns, dtype=np.float64).ravel()
    x_tsmom = np.asarray(tsmom_returns, dtype=np.float64).ravel()

    n = len(y)
    if n == 0:
        raise ValueError("strategy_returns must not be empty")
    if len(x_tsmom) != n:
        raise ValueError("tsmom_returns must have the same length as strategy_returns")

    has_carry = carry_returns is not None
    if has_carry:
        x_carry = np.asarray(carry_returns, dtype=np.float64).ravel()
        if len(x_carry) != n:
            raise ValueError("carry_returns must have the same length as strategy_returns")
        X = np.column_stack([np.ones(n), x_tsmom, x_carry])
    else:
        x_carry = np.zeros(n)
        X = np.column_stack([np.ones(n), x_tsmom])

    # OLS via numpy lstsq
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    alpha_daily = coeffs[0]
    beta_tsmom = coeffs[1]
    beta_carry = coeffs[2] if has_carry else 0.0

    # Annualized values
    independent_alpha = alpha_daily * annualize
    total_return_ann = float(np.mean(y)) * annualize
    tsmom_beta_return = float(beta_tsmom * np.mean(x_tsmom)) * annualize
    carry_beta_return = float(beta_carry * np.mean(x_carry)) * annualize if has_carry else 0.0

    # R-squared
    y_hat = X @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    # Percentage decomposition
    components_sum = abs(tsmom_beta_return) + abs(carry_beta_return) + abs(independent_alpha)
    if components_sum > 0.0:
        tsmom_pct = abs(tsmom_beta_return) / components_sum * 100.0
        carry_pct = abs(carry_beta_return) / components_sum * 100.0
        alpha_pct = abs(independent_alpha) / components_sum * 100.0
    else:
        tsmom_pct = 0.0
        carry_pct = 0.0
        alpha_pct = 0.0

    return BaselineDecomposition(
        total_return_ann=float(total_return_ann),
        tsmom_beta_return=float(tsmom_beta_return),
        carry_beta_return=float(carry_beta_return),
        independent_alpha=float(independent_alpha),
        tsmom_pct=float(tsmom_pct),
        carry_pct=float(carry_pct),
        alpha_pct=float(alpha_pct),
        r_squared=float(r_squared),
    )
