"""Signal Blending Layer.

Blends multiple strategy signals into a single net signal with:
- Weighted signal aggregation
- Fundamental direction filtering
- Volatility targeting scale adjustment
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class BlendedSignal:
    """Result of blending multiple strategy signals.

    Attributes
    ----------
    net_signal : float
        Weighted sum of strategy signals, clipped to [-1, 1].
    strategy_weights : dict[str, float]
        Weights used for blending.
    active_strategies : tuple[str, ...]
        Names of strategies that contributed to the blend.
    """

    net_signal: float
    strategy_weights: dict[str, float]
    active_strategies: tuple[str, ...]


def blend_signals(
    signals: dict[str, float],
    weights: dict[str, float],
) -> BlendedSignal:
    """Blend multiple strategy signals into a single net signal.

    Computes ``net_signal = sum(w[i] * s[i])`` for strategies present in both
    *signals* and *weights*, then clips the result to [-1, 1].

    Parameters
    ----------
    signals : dict[str, float]
        Mapping of strategy name to signal value.
    weights : dict[str, float]
        Mapping of strategy name to weight.

    Returns
    -------
    BlendedSignal
        The blended result.
    """
    active = sorted(set(signals) & set(weights))
    if not active:
        return BlendedSignal(
            net_signal=0.0,
            strategy_weights={},
            active_strategies=(),
        )

    weight_sum = sum(weights[s] for s in active)
    if weight_sum > 0:
        normalized_weights = {s: weights[s] / weight_sum for s in active}
    else:
        normalized_weights = {s: weights[s] for s in active}

    raw = sum(normalized_weights[s] * signals[s] for s in active)
    net = float(np.clip(raw, -1.0, 1.0))

    return BlendedSignal(
        net_signal=net,
        strategy_weights=normalized_weights,
        active_strategies=tuple(active),
    )


def apply_direction_filter(signal: float, direction: str) -> float:
    """Apply fundamental direction constraint to a signal.

    Parameters
    ----------
    signal : float
        Raw net signal.
    direction : str
        One of ``"long"``, ``"short"``, ``"neutral"``.

    Returns
    -------
    float
        Filtered signal: long -> max(0, s), short -> min(0, s), neutral -> s.

    Raises
    ------
    ValueError
        If *direction* is not one of the allowed values.
    """
    if direction == "long":
        return max(0.0, signal)
    if direction == "short":
        return min(0.0, signal)
    if direction == "neutral":
        return signal
    raise ValueError(f"Unknown direction: {direction!r}. Expected 'long', 'short', or 'neutral'.")


def apply_vol_targeting(
    signal: float,
    target_vol: float,
    realized_vol: float,
    clip_low: float = 0.2,
    clip_high: float = 3.0,
) -> float:
    """Scale signal by target_vol / realized_vol, clipped.

    Parameters
    ----------
    signal : float
        The net signal to scale.
    target_vol : float
        Target annualised volatility (e.g. 0.15).
    realized_vol : float
        Realised annualised volatility.
    clip_low : float
        Minimum allowed scaling factor.
    clip_high : float
        Maximum allowed scaling factor.

    Returns
    -------
    float
        Scaled signal, clipped to [-1, 1] after scaling.
    """
    if realized_vol <= 0.0:
        return signal

    scale = target_vol / realized_vol
    scale = float(np.clip(scale, clip_low, clip_high))
    scaled = signal * scale
    return float(np.clip(scaled, -1.0, 1.0))
