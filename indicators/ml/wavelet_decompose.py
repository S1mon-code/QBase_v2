import numpy as np


def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean with NaN-padding at start.  NaN-safe."""
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window or window < 1:
        return out
    # Handle NaN in input by using a simple loop for correctness
    buf = 0.0
    count = 0
    for i in range(n):
        if np.isnan(data[i]):
            # reset accumulator — need contiguous valid values
            buf = 0.0
            count = 0
            continue
        buf += data[i]
        count += 1
        if count > window:
            buf -= data[i - window]
            count = window
        if count == window:
            out[i] = buf / window
    return out


def wavelet_features(
    closes: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wavelet decomposition into trend + detail components.

    Uses a cascade of simple moving average filters as a wavelet-free
    approximation: each level doubles the smoothing window, separating
    low-frequency trend from high-frequency noise.

    Parameters
    ----------
    closes : (N,) price series.
    wavelet : ignored (kept for API compatibility).
    level : number of decomposition levels (each doubles the window).

    Returns
    -------
    trend_component : (N,) low-frequency (smoothed) price.
    detail_component : (N,) high-frequency residual (closes - trend).
    energy_ratio : (N,) ratio of detail energy to total energy over a
        rolling window, indicating the noise level (0 = pure trend,
        1 = pure noise).
    """
    n = len(closes)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    # Cascade of SMA filters mimics multi-resolution decomposition
    smooth = closes.copy().astype(np.float64)
    for lev in range(level):
        win = 2 ** (lev + 1)  # 2, 4, 8, 16 for level=4
        smooth = _rolling_mean(smooth, win)

    trend_component = smooth
    detail_component = closes - trend_component

    # Rolling energy ratio over a window
    energy_window = 2 ** level
    energy_ratio = np.full(n, np.nan, dtype=np.float64)

    for i in range(energy_window - 1, n):
        if np.isnan(trend_component[i]):
            continue
        start = i - energy_window + 1
        det_seg = detail_component[start : i + 1]
        full_seg = closes[start : i + 1]
        if np.any(np.isnan(det_seg)) or np.any(np.isnan(full_seg)):
            continue
        total_energy = np.sum(full_seg ** 2)
        detail_energy = np.sum(det_seg ** 2)
        if total_energy > 1e-12:
            energy_ratio[i] = detail_energy / total_energy
        else:
            energy_ratio[i] = 0.0

    return trend_component, detail_component, energy_ratio
