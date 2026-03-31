"""Parameter robustness analysis: plateau detection + multi-seed validation."""

from __future__ import annotations

import random
import statistics
from typing import Any, Callable

from optimizer.config import DEFAULT_SEEDS, NARROW_RADIUS, ROBUSTNESS_THRESHOLD


def _perturb_params(
    params: dict[str, Any],
    radius: float,
    rng: random.Random,
) -> dict[str, Any]:
    """Create a neighbor by perturbing each param within ±radius fraction.

    Integer params are rounded. Values stay within [original * (1-radius), original * (1+radius)].
    """
    neighbor: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            if v == 0:
                low, high = -radius, radius
            else:
                a = v * (1.0 - radius)
                b = v * (1.0 + radius)
                low, high = min(a, b), max(a, b)
            new_val = rng.uniform(low, high)
            if isinstance(v, int):
                new_val = round(new_val)
            neighbor[k] = type(v)(new_val)
        else:
            neighbor[k] = v
    return neighbor


def check_robustness(
    best_params: dict[str, Any],
    best_score: float,
    evaluate_fn: Callable[[dict[str, Any]], float],
    n_samples: int | None = None,
    radius: float = NARROW_RADIUS,
) -> dict[str, Any]:
    """Sample neighbors in ±radius around best_params and check for plateau.

    A parameter set is considered ROBUST (plateau) if >=60% of neighbors
    score above best_score * 0.5.

    Args:
        best_params: Best parameter set found by optimizer.
        best_score: Score of best parameter set.
        evaluate_fn: Callable that takes params dict and returns score.
        n_samples: Number of neighbor samples. Default: max(20, n_params*5).
        radius: Perturbation radius as fraction (default 0.15 = ±15%).

    Returns:
        {
            "is_robust": bool,
            "neighbor_scores": list[float],
            "above_threshold_pct": float,
            "neighbor_mean": float,
            "neighbor_std": float,
        }
    """
    n_params = len(best_params)
    if n_samples is None:
        n_samples = max(20, n_params * 5)

    rng = random.Random(42)
    threshold = best_score * 0.5
    neighbor_scores: list[float] = []

    for _ in range(n_samples):
        neighbor = _perturb_params(best_params, radius, rng)
        score = evaluate_fn(neighbor)
        neighbor_scores.append(score)

    above_count = sum(1 for s in neighbor_scores if s > threshold)
    above_pct = above_count / len(neighbor_scores) if neighbor_scores else 0.0

    return {
        "is_robust": above_pct >= ROBUSTNESS_THRESHOLD,
        "neighbor_scores": neighbor_scores,
        "above_threshold_pct": above_pct,
        "neighbor_mean": statistics.mean(neighbor_scores) if neighbor_scores else 0.0,
        "neighbor_std": statistics.stdev(neighbor_scores) if len(neighbor_scores) > 1 else 0.0,
    }


def multi_seed_optimize(
    optimize_fn: Callable[[int], tuple[dict, float]],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Run optimization with multiple seeds, take median result.

    Args:
        optimize_fn: Callable(seed) -> (params, score).
        seeds: Tuple of random seeds to use.

    Returns:
        {
            "best_params": dict,   # params from median-scoring run
            "best_score": float,   # median score
            "is_consistent": bool, # std < 50% * mean
            "all_scores": list[float],
            "all_params": list[dict],
        }
    """
    all_scores: list[float] = []
    all_params: list[dict] = []

    for seed in seeds:
        params, score = optimize_fn(seed)
        all_scores.append(score)
        all_params.append(params)

    # Find median result
    sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i])
    median_idx = sorted_indices[len(sorted_indices) // 2]

    mean_score = statistics.mean(all_scores) if all_scores else 0.0
    std_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0

    # Consistent if std < 50% of mean (low variance across seeds)
    is_consistent = (std_score < 0.5 * abs(mean_score)) if mean_score != 0 else (std_score == 0)

    return {
        "best_params": all_params[median_idx],
        "best_score": all_scores[median_idx],
        "is_consistent": is_consistent,
        "all_scores": all_scores,
        "all_params": all_params,
    }
