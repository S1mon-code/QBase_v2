"""Shared helper functions for indicators. Avoids duplication across files."""

import numpy as np


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. First `period-1` values are np.nan."""
    result = np.full(arr.size, np.nan)
    if arr.size < period:
        return result

    result[period - 1] = arr[:period].mean()
    k = 2.0 / (period + 1.0)
    for i in range(period, arr.size):
        result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


def _ema_no_warmup(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA seeded with first value, no NaN warmup. For cascaded EMA use (DEMA/TEMA/T3)."""
    if arr.size == 0:
        return np.array([], dtype=float)
    out = np.empty(arr.size)
    out[0] = arr[0]
    k = 2.0 / (period + 1.0)
    for i in range(1, arr.size):
        if np.isnan(arr[i]):
            out[i] = out[i - 1]
        else:
            out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def _ema_skip_nan(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA that skips NaN values, seeding from first `period` valid values."""
    result = np.full(arr.size, np.nan)
    if arr.size < period:
        return result

    valid_count = 0
    seed_sum = 0.0
    seed_idx = -1
    for i in range(arr.size):
        if not np.isnan(arr[i]):
            valid_count += 1
            seed_sum += arr[i]
            if valid_count == period:
                seed_idx = i
                break

    if seed_idx < 0:
        return result

    result[seed_idx] = seed_sum / period
    k = 2.0 / (period + 1.0)
    for i in range(seed_idx + 1, arr.size):
        if np.isnan(arr[i]):
            result[i] = result[i - 1]
        else:
            result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average. First `period-1` values are np.nan."""
    result = np.full(arr.size, np.nan)
    if arr.size < period:
        return result

    cumsum = np.cumsum(arr)
    result[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0.0], cumsum[:-period]))) / period
    return result


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI calculation (Wilder's smoothing). First `period` values are np.nan."""
    n = closes.size
    result = np.full(n, np.nan)
    if n <= period:
        return result

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    if avg_loss == 0:
        result[period] = 100.0
    else:
        result[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return result
