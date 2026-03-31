"""Weight Calculation — 3-stage progressive weighting.

Stage 1: Equal Weight (< 5 strategies)
Stage 2: Inverse Volatility (5-10 strategies)
Stage 3: HRP x Alpha x Consistency (mature)

Uses Ledoit-Wolf shrinkage for covariance estimation in HRP.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def equal_weights(strategies: list[str]) -> dict[str, float]:
    """Stage 1: Equal weight across all strategies.

    Parameters
    ----------
    strategies : list[str]
        List of strategy names.

    Returns
    -------
    dict[str, float]
        Equal weights summing to 1.0.
    """
    if not strategies:
        return {}
    w = 1.0 / len(strategies)
    return {s: w for s in strategies}


def inverse_volatility_weights(
    strategy_vols: dict[str, float],
) -> dict[str, float]:
    """Stage 2: Inverse volatility weighting.

    Parameters
    ----------
    strategy_vols : dict[str, float]
        Mapping of strategy name to annualised volatility.

    Returns
    -------
    dict[str, float]
        Weights proportional to 1/vol, normalised to sum to 1.0.
    """
    if not strategy_vols:
        return {}

    inv = {s: 1.0 / v for s, v in strategy_vols.items() if v > 0.0}
    if not inv:
        return equal_weights(list(strategy_vols.keys()))

    total = sum(inv.values())
    return {s: w / total for s, w in inv.items()}


def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Compute distance matrix from correlation: sqrt(0.5 * (1 - corr)).

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix (n x n).

    Returns
    -------
    np.ndarray
        Distance matrix (n x n).
    """
    return np.sqrt(0.5 * (1.0 - corr))


def _quasi_diagonal_sort(link: np.ndarray, n: int) -> list[int]:
    """Reorder items into quasi-diagonal form from linkage matrix.

    Parameters
    ----------
    link : np.ndarray
        Linkage matrix from scipy.cluster.hierarchy.linkage.
    n : int
        Number of original items.

    Returns
    -------
    list[int]
        Sorted indices.
    """
    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]

    while max(sort_ix) >= n:
        new_sort_ix: list[int] = []
        for ix in sort_ix:
            if ix >= n:
                row = int(ix - n)
                new_sort_ix.append(int(link[row, 0]))
                new_sort_ix.append(int(link[row, 1]))
            else:
                new_sort_ix.append(ix)
        sort_ix = new_sort_ix

    return sort_ix


def _recursive_bisection(cov: np.ndarray, sorted_ix: list[int]) -> np.ndarray:
    """Allocate weights via recursive bisection with inverse variance.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    sorted_ix : list[int]
        Quasi-diagonally sorted indices.

    Returns
    -------
    np.ndarray
        Weight array of length n.
    """
    n = cov.shape[0]
    weights = np.ones(n)
    clusters = [sorted_ix]

    while clusters:
        new_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Inverse variance for each half
            left_cov = cov[np.ix_(left, left)]
            right_cov = cov[np.ix_(right, right)]

            left_inv_var = 1.0 / np.diag(left_cov).sum() if np.diag(left_cov).sum() > 0 else 1.0
            right_inv_var = 1.0 / np.diag(right_cov).sum() if np.diag(right_cov).sum() > 0 else 1.0

            alloc_left = left_inv_var / (left_inv_var + right_inv_var)
            alloc_right = 1.0 - alloc_left

            for ix in left:
                weights[ix] *= alloc_left
            for ix in right:
                weights[ix] *= alloc_right

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        clusters = new_clusters

    return weights


def hrp_weights(returns_matrix: np.ndarray, strategy_names: list[str]) -> dict[str, float]:
    """Hierarchical Risk Parity (Lopez de Prado).

    Uses Ledoit-Wolf shrinkage for covariance estimation.

    Steps:
        1. Correlation distance matrix
        2. Single linkage hierarchical clustering
        3. Quasi-diagonal reordering
        4. Recursive bisection with inverse variance allocation

    Parameters
    ----------
    returns_matrix : np.ndarray
        Returns matrix of shape (n_periods, n_strategies).
    strategy_names : list[str]
        Strategy names corresponding to columns.

    Returns
    -------
    dict[str, float]
        Weights summing to 1.0.
    """
    n = returns_matrix.shape[1]
    if n == 0:
        return {}
    if n == 1:
        return {strategy_names[0]: 1.0}

    # Ledoit-Wolf shrinkage covariance
    lw = LedoitWolf()
    lw.fit(returns_matrix)
    cov = lw.covariance_

    # Correlation from covariance
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1e-10
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)

    # Distance matrix
    dist = _correlation_distance(corr)
    np.fill_diagonal(dist, 0.0)

    # Condensed distance for scipy linkage
    condensed = squareform(dist, checks=False)

    # Single linkage clustering
    link = linkage(condensed, method="single")

    # Quasi-diagonal sort
    sorted_ix = _quasi_diagonal_sort(link, n)

    # Recursive bisection
    raw_w = _recursive_bisection(cov, sorted_ix)

    # Normalise
    total = raw_w.sum()
    if total > 0:
        raw_w = raw_w / total

    return {strategy_names[i]: float(raw_w[i]) for i in range(n)}


def alpha_adjusted_weights(
    hrp_w: dict[str, float],
    independent_alphas: dict[str, float],
    wf_win_rates: dict[str, float],
    min_alpha_factor: float = 0.2,
) -> dict[str, float]:
    """Stage 3: HRP x alpha_factor x consistency_factor, normalised.

    Parameters
    ----------
    hrp_w : dict[str, float]
        Base HRP weights.
    independent_alphas : dict[str, float]
        Independent alpha per strategy (from attribution).
    wf_win_rates : dict[str, float]
        Walk-forward window win rate per strategy.
    min_alpha_factor : float
        Minimum alpha scaling factor.

    Returns
    -------
    dict[str, float]
        Adjusted weights normalised to sum to 1.0.
    """
    if not hrp_w:
        return {}

    common = sorted(set(hrp_w) & set(independent_alphas) & set(wf_win_rates))
    if not common:
        return hrp_w

    max_alpha = max(independent_alphas[s] for s in common) if common else 1.0
    if max_alpha <= 0:
        max_alpha = 1.0

    adjusted: dict[str, float] = {}
    for s in common:
        alpha_factor = max(min_alpha_factor, independent_alphas[s] / max_alpha)
        consistency_factor = wf_win_rates[s]
        adjusted[s] = hrp_w[s] * alpha_factor * consistency_factor

    total = sum(adjusted.values())
    if total <= 0:
        return equal_weights(common)

    return {s: w / total for s, w in adjusted.items()}


def clip_and_redistribute(
    weights: dict[str, float],
    max_weight: float = 0.25,
) -> dict[str, float]:
    """Clip any weight above max_weight, redistribute excess proportionally.

    Parameters
    ----------
    weights : dict[str, float]
        Input weights (should sum to ~1.0).
    max_weight : float
        Maximum allowed weight per strategy.

    Returns
    -------
    dict[str, float]
        Clipped and redistributed weights summing to 1.0.
    """
    if not weights:
        return {}

    result = dict(weights)

    for _ in range(10):  # iterate to convergence
        excess = 0.0
        below: dict[str, float] = {}

        for s, w in result.items():
            if w > max_weight:
                excess += w - max_weight
                result[s] = max_weight
            else:
                below[s] = w

        if excess <= 1e-12:
            break

        below_total = sum(below.values())
        if below_total <= 0:
            break

        for s in below:
            result[s] += excess * (below[s] / below_total)

    # Final normalisation
    total = sum(result.values())
    if total > 0:
        result = {s: w / total for s, w in result.items()}

    return result
