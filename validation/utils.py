"""Shared utilities for validation modules."""
import numpy as np


def compute_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio from daily returns (rf=0)."""
    if len(returns) == 0:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0.0:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(252))
