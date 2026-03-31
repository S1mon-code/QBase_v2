import numpy as np
from scipy.signal import hilbert


def hilbert_transform_features(
    closes: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hilbert transform: instantaneous phase and amplitude.

    Applies the Hilbert transform on rolling detrended windows to extract
    the dominant cycle's amplitude, phase, and instantaneous frequency.

    Parameters
    ----------
    closes : (N,) price series.
    period : rolling window length for detrending and transform.

    Returns
    -------
    amplitude : (N,) instantaneous amplitude of the dominant cycle.
    phase : (N,) instantaneous phase in radians (-pi to pi).
    instantaneous_freq : (N,) instantaneous frequency (cycles per bar).
    """
    n = len(closes)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    amp = np.full(n, np.nan, dtype=np.float64)
    phs = np.full(n, np.nan, dtype=np.float64)
    freq = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return amp, phs, freq

    for i in range(period, n):
        window = closes[i - period : i].astype(np.float64)
        if np.any(np.isnan(window)):
            continue

        # Detrend: remove linear trend to improve edge behaviour
        x = np.arange(period, dtype=np.float64)
        slope, intercept = np.polyfit(x, window, 1)
        detrended = window - (slope * x + intercept)

        # Hilbert transform
        analytic = hilbert(detrended)
        inst_amp = np.abs(analytic)
        inst_phase = np.angle(analytic)

        # Use last value in window as the current bar's reading
        amp[i] = inst_amp[-1]
        phs[i] = inst_phase[-1]

        # Instantaneous frequency from phase derivative
        if period >= 2:
            dp = np.diff(np.unwrap(inst_phase))
            freq[i] = dp[-1] / (2.0 * np.pi)

    return amp, phs, freq
