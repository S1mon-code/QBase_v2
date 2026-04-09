"""Signal Blending Layer — Carver/Man AHL Standard.

Implements the full forecast combination pipeline:
    1. Forecast Scaling — normalize each strategy so avg|forecast| = 10
    2. Forecast Capping — clip to [-20, +20]
    3. Weighted Combination — w_i × f_i with NaN handling
    4. FDM Application — Forecast Diversification Multiplier
    5. Re-Capping — final clip to [-20, +20]
    6. Direction Filter — long/short/neutral constraint
    7. Position Sizing — forecast/10 × vol-targeted lots

Forecast scale convention:
    0   = no signal (flat)
    10  = average confidence (full position)
    20  = maximum confidence (2x position)
    -20 = maximum short confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

FORECAST_CAP: float = 20.0
FORECAST_TARGET_ABS: float = 10.0
FDM_CAP: float = 2.5
MIN_COVERAGE_RATIO: float = 0.25  # min fraction of strategies with signal


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ForecastScalar:
    """Per-strategy forecast scalar so that avg|forecast| = 10.

    Attributes
    ----------
    strategy : str
        Strategy name.
    scalar : float
        Multiplicative scalar: ``scaled = raw × scalar``.
    raw_mean_abs : float
        Mean absolute raw forecast used to compute the scalar.
    """

    strategy: str
    scalar: float
    raw_mean_abs: float


@dataclass(frozen=True)
class BlendedForecast:
    """Result of the full forecast combination pipeline.

    Attributes
    ----------
    forecast : float
        Final combined forecast in [-20, +20] (after FDM and re-cap).
    raw_combined : float
        Weighted sum before FDM.
    fdm : float
        Forecast Diversification Multiplier applied.
    active_strategies : tuple[str, ...]
        Strategies that contributed (had non-NaN forecasts).
    coverage : float
        Fraction of total strategies that had valid forecasts.
    weights_used : dict[str, float]
        Re-normalized weights for active strategies.
    """

    forecast: float
    raw_combined: float
    fdm: float
    active_strategies: tuple[str, ...]
    coverage: float
    weights_used: dict[str, float]


# ── Forecast Scaling ──────────────────────────────────────────────────────────


def compute_forecast_scalar(
    raw_forecasts: np.ndarray,
    target_abs: float = FORECAST_TARGET_ABS,
    min_obs: int = 50,
) -> float:
    """Compute the scalar that maps raw forecasts so avg|f| = target_abs.

    Parameters
    ----------
    raw_forecasts : np.ndarray
        Historical raw forecast values (1-D).
    target_abs : float
        Target average absolute forecast (default 10).
    min_obs : int
        Minimum observations required; returns 1.0 if insufficient.

    Returns
    -------
    float
        Forecast scalar.
    """
    valid = raw_forecasts[~np.isnan(raw_forecasts)]
    if len(valid) < min_obs:
        return 1.0

    mean_abs = float(np.mean(np.abs(valid)))
    if mean_abs <= 1e-10:
        return 1.0

    return target_abs / mean_abs


def scale_and_cap(
    forecast: float,
    scalar: float,
    cap: float = FORECAST_CAP,
) -> float:
    """Scale a raw forecast and cap to [-cap, +cap].

    Parameters
    ----------
    forecast : float
        Raw forecast value.
    scalar : float
        Forecast scalar (from ``compute_forecast_scalar``).
    cap : float
        Maximum absolute forecast (default 20).

    Returns
    -------
    float
        Scaled and capped forecast.
    """
    if np.isnan(forecast):
        return float("nan")
    return float(np.clip(forecast * scalar, -cap, cap))


# ── Forecast Diversification Multiplier ───────────────────────────────────────


def compute_fdm(
    weights: dict[str, float],
    correlation_matrix: np.ndarray,
    strategy_order: list[str],
    cap: float = FDM_CAP,
) -> float:
    """Compute Forecast Diversification Multiplier.

    FDM = 1 / sqrt(w' C w), capped at *cap*.

    Parameters
    ----------
    weights : dict[str, float]
        Strategy weights (must sum to ~1.0).
    correlation_matrix : np.ndarray
        Correlation matrix of forecasts, shape (n, n).
    strategy_order : list[str]
        Strategy names corresponding to rows/columns of *correlation_matrix*.
    cap : float
        Maximum FDM (default 2.5).

    Returns
    -------
    float
        FDM value in [1.0, cap].
    """
    n = len(strategy_order)
    if n <= 1:
        return 1.0

    w = np.array([weights.get(s, 0.0) for s in strategy_order])
    w_sum = w.sum()
    if w_sum <= 0:
        return 1.0
    w = w / w_sum

    wcw = float(w @ correlation_matrix @ w)
    if wcw <= 0:
        return 1.0

    fdm = 1.0 / np.sqrt(wcw)
    return float(min(fdm, cap))


def estimate_forecast_correlation(
    forecast_matrix: np.ndarray,
    shrinkage: float = 0.5,
) -> np.ndarray:
    """Estimate correlation matrix of forecasts with shrinkage.

    Uses ``C_shrunk = δ × C_sample + (1-δ) × C_prior`` where C_prior has
    all off-diagonal elements equal to the average sample correlation.

    Parameters
    ----------
    forecast_matrix : np.ndarray
        Shape (n_obs, n_strategies).  NaN values are handled by pairwise
        complete observations.
    shrinkage : float
        Shrinkage parameter δ in [0, 1].  0 = full prior, 1 = full sample.

    Returns
    -------
    np.ndarray
        Shrunk correlation matrix, shape (n_strategies, n_strategies).
    """
    n = forecast_matrix.shape[1]
    if n <= 1:
        return np.ones((1, 1))

    # Pairwise correlation (handles NaN via masked computation)
    sample_corr = np.full((n, n), 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            mask = ~np.isnan(forecast_matrix[:, i]) & ~np.isnan(forecast_matrix[:, j])
            if mask.sum() < 20:
                sample_corr[i, j] = sample_corr[j, i] = 0.5  # prior assumption
            else:
                c = np.corrcoef(forecast_matrix[mask, i], forecast_matrix[mask, j])[0, 1]
                if np.isnan(c):
                    c = 0.5
                sample_corr[i, j] = sample_corr[j, i] = c

    # Prior: average correlation on off-diagonals
    off_diag = sample_corr[np.triu_indices(n, k=1)]
    avg_corr = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.5
    prior_corr = np.full((n, n), avg_corr)
    np.fill_diagonal(prior_corr, 1.0)

    # Shrink
    shrunk = shrinkage * sample_corr + (1.0 - shrinkage) * prior_corr
    return shrunk


# ── Core Blending ─────────────────────────────────────────────────────────────


def blend_forecasts(
    forecasts: dict[str, float],
    weights: dict[str, float],
    fdm: float = 1.0,
    total_strategies: int | None = None,
    min_coverage: float = MIN_COVERAGE_RATIO,
    cap: float = FORECAST_CAP,
) -> BlendedForecast:
    """Combine multiple scaled/capped forecasts into one net forecast.

    Pipeline:
        1. Filter NaN forecasts, re-normalize weights
        2. Weighted sum
        3. Apply FDM
        4. Coverage scaling (if < min_coverage strategies available)
        5. Re-cap to [-cap, +cap]

    Parameters
    ----------
    forecasts : dict[str, float]
        Scaled & capped forecasts per strategy (NaN = missing).
    weights : dict[str, float]
        Strategy weights (should sum to ~1.0 when all present).
    fdm : float
        Forecast Diversification Multiplier.
    total_strategies : int | None
        Total number of strategies in the portfolio (for coverage ratio).
        If None, uses ``len(weights)``.
    min_coverage : float
        Minimum coverage ratio (default 0.25).  If fewer strategies have
        valid signals, the combined forecast is scaled down.
    cap : float
        Forecast cap (default 20).

    Returns
    -------
    BlendedForecast
        Combined forecast with metadata.
    """
    if total_strategies is None:
        total_strategies = len(weights)

    # Filter available (non-NaN) forecasts
    available = {
        k: v for k, v in forecasts.items()
        if k in weights and not np.isnan(v)
    }

    if not available:
        return BlendedForecast(
            forecast=0.0, raw_combined=0.0, fdm=fdm,
            active_strategies=(), coverage=0.0, weights_used={},
        )

    # Re-normalize weights for available strategies
    total_w = sum(weights[k] for k in available)
    if total_w <= 0:
        renorm = {k: 1.0 / len(available) for k in available}
    else:
        renorm = {k: weights[k] / total_w for k in available}

    # Weighted combination
    raw_combined = sum(renorm[k] * available[k] for k in available)

    # Apply FDM
    adjusted = fdm * raw_combined

    # Coverage scaling
    coverage = len(available) / max(total_strategies, 1)
    if coverage < min_coverage:
        adjusted *= coverage / min_coverage

    # Re-cap
    final = float(np.clip(adjusted, -cap, cap))

    return BlendedForecast(
        forecast=final,
        raw_combined=float(raw_combined),
        fdm=fdm,
        active_strategies=tuple(sorted(available.keys())),
        coverage=coverage,
        weights_used=renorm,
    )


# ── Direction Filter ──────────────────────────────────────────────────────────


def apply_direction_filter(forecast: float, direction: str) -> float:
    """Apply fundamental direction constraint to a forecast.

    Parameters
    ----------
    forecast : float
        Combined forecast in [-20, +20].
    direction : str
        ``"long"`` → max(0, f), ``"short"`` → min(0, f), ``"neutral"`` → f.

    Returns
    -------
    float
        Direction-filtered forecast.
    """
    if direction == "long":
        return max(0.0, forecast)
    if direction == "short":
        return min(0.0, forecast)
    if direction == "neutral":
        return forecast
    raise ValueError(f"Unknown direction: {direction!r}")


# ── Position Sizing from Forecast ─────────────────────────────────────────────


def forecast_to_position(
    forecast: float,
    capital: float,
    target_vol: float,
    price: float,
    multiplier: float,
    ann_vol: float,
    margin_rate: float = 0.12,
    max_margin_util: float = 0.80,
) -> int:
    """Convert a forecast in [-20, +20] to a position in contracts.

    Formula:
        lots = (forecast / 10) × (capital × target_vol) / (price × multiplier × ann_vol)

    Parameters
    ----------
    forecast : float
        Combined forecast in [-20, +20].
    capital : float
        Current portfolio equity.
    target_vol : float
        Target annualized volatility (e.g. 0.15).
    price : float
        Current instrument price.
    multiplier : float
        Contract multiplier (e.g. 100 for iron ore).
    ann_vol : float
        Annualized instrument volatility.
    margin_rate : float
        Margin rate for the instrument.
    max_margin_util : float
        Maximum margin as fraction of capital.

    Returns
    -------
    int
        Target position in contracts (positive = long, negative = short).
    """
    if price <= 0 or ann_vol <= 1e-8 or capital <= 0:
        return 0

    risk_per_contract = price * multiplier * ann_vol
    ideal = (forecast / FORECAST_TARGET_ABS) * (capital * target_vol) / risk_per_contract

    # Margin cap
    margin_per_lot = price * multiplier * margin_rate
    if margin_per_lot > 0:
        max_lots = capital * max_margin_util / margin_per_lot
        ideal = np.clip(ideal, -max_lots, max_lots)

    return int(ideal)


def portfolio_vol_cap(
    total_lots: np.ndarray,
    closes: np.ndarray,
    multiplier: float,
    capital: float,
    target_vol: float,
    max_vol_ratio: float = 1.5,
    lookback: int = 20,
) -> np.ndarray:
    """Apply portfolio-level volatility cap.

    If realized portfolio vol exceeds ``target_vol × max_vol_ratio``,
    scale down all positions proportionally.

    Parameters
    ----------
    total_lots : np.ndarray
        Target position in lots per bar.
    closes : np.ndarray
        Price array (same length as total_lots).
    multiplier : float
        Contract multiplier.
    capital : float
        Portfolio capital.
    target_vol : float
        Target annualized volatility.
    max_vol_ratio : float
        Maximum allowed ratio of realized/target vol (default 1.5x).
    lookback : int
        Window for realized vol estimation.

    Returns
    -------
    np.ndarray
        Adjusted position array with vol cap applied.
    """
    result = total_lots.copy()
    n = len(total_lots)

    for i in range(lookback, n):
        if total_lots[i] == 0 or closes[i] <= 0:
            continue

        # Estimate portfolio daily P&L volatility
        # position_value = lots × multiplier × price
        # daily_pnl_vol ≈ position_value × daily_price_vol
        start = max(0, i - lookback)
        window = closes[start:i + 1]
        if len(window) < 5:
            continue

        rets = np.diff(window) / window[:-1]
        daily_vol = float(np.std(rets))
        if daily_vol <= 1e-8:
            continue

        ann_vol = daily_vol * np.sqrt(252)
        position_value = abs(total_lots[i]) * multiplier * closes[i]
        portfolio_vol = (position_value * ann_vol) / capital

        if portfolio_vol > target_vol * max_vol_ratio:
            scale = (target_vol * max_vol_ratio) / portfolio_vol
            result[i] = int(total_lots[i] * scale)

    return result


def monitor_correlation_regime(
    returns_matrix: np.ndarray,
    strategy_names: list[str],
    lookback: int = 60,
    warning_threshold: float = 0.7,
    crisis_threshold: float = 0.85,
) -> dict[str, Any]:
    """Monitor rolling average correlation between strategies.

    When correlations spike (crisis), diversification benefit drops and
    portfolio risk increases. Returns warnings and suggested position scaling.

    Parameters
    ----------
    returns_matrix : np.ndarray
        Shape (n_obs, n_strategies).
    strategy_names : list[str]
        Strategy names.
    lookback : int
        Rolling window for correlation estimation.
    warning_threshold : float
        Average correlation above this triggers WARNING.
    crisis_threshold : float
        Average correlation above this triggers CRISIS (reduce positions).

    Returns
    -------
    dict with keys:
        avg_correlation : float
        regime : str ("normal", "warning", "crisis")
        position_scale : float (1.0 = normal, 0.5 = crisis)
        pair_correlations : dict
    """
    n = returns_matrix.shape[1]
    if n < 2:
        return {
            "avg_correlation": 0.0, "regime": "normal",
            "position_scale": 1.0, "pair_correlations": {},
        }

    # Use last `lookback` observations
    recent = returns_matrix[-lookback:] if len(returns_matrix) > lookback else returns_matrix
    corr = np.corrcoef(recent.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Average off-diagonal correlation
    off_diag = []
    pair_corrs = {}
    for i in range(n):
        for j in range(i + 1, n):
            c = float(corr[i, j])
            off_diag.append(c)
            pair_corrs[f"{strategy_names[i]}_vs_{strategy_names[j]}"] = round(c, 3)

    avg_corr = float(np.mean(off_diag)) if off_diag else 0.0

    if avg_corr >= crisis_threshold:
        regime = "crisis"
        scale = 0.5
    elif avg_corr >= warning_threshold:
        regime = "warning"
        scale = 0.75
    else:
        regime = "normal"
        scale = 1.0

    return {
        "avg_correlation": round(avg_corr, 3),
        "regime": regime,
        "position_scale": scale,
        "pair_correlations": pair_corrs,
    }


def should_rebalance(
    current_pos: int,
    target_pos: int,
    buffer_fraction: float = 0.10,
) -> bool:
    """Check if position should be rebalanced (Carver buffer zone).

    Only rebalance if the deviation from target exceeds *buffer_fraction*
    of the target position.  Always rebalance on direction change or
    flat → position transitions.

    Parameters
    ----------
    current_pos : int
        Current position in contracts.
    target_pos : int
        Target position from ``forecast_to_position``.
    buffer_fraction : float
        Buffer as fraction of target (default 10%).

    Returns
    -------
    bool
        Whether to adjust position.
    """
    # Always rebalance on flat ↔ position or direction flip
    if current_pos == 0 and target_pos != 0:
        return True
    if target_pos == 0 and current_pos != 0:
        return True
    if current_pos != 0 and target_pos != 0 and (current_pos > 0) != (target_pos > 0):
        return True

    # Same direction — check buffer
    if target_pos == 0:
        return False
    deviation = abs(current_pos - target_pos) / max(abs(target_pos), 1)
    return deviation > buffer_fraction
