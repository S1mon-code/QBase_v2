"""Fourier seasonal decomposition — extracts dominant seasonal cycle and measures its explanatory strength."""
import numpy as np


def seasonal_strength(closes: np.ndarray, period: int = 252) -> tuple:
    """Fourier-based seasonal component extraction.

    Extracts the dominant seasonal cycle from the price series using
    discrete Fourier transform, then measures how much variance the
    seasonal component explains.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    period : int
        Expected seasonal period (252 = annual for daily bars).

    Returns
    -------
    seasonal_component : np.ndarray
        Extracted seasonal component (same length as input, NaN padded).
    seasonal_strength_ratio : np.ndarray
        variance(seasonal) / variance(total). Range 0-1.
        Higher = stronger seasonality.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan)

    seasonal_component = np.full(n, np.nan)
    strength_ratio = np.full(n, np.nan)

    # Log returns for stationarity
    log_prices = np.log(np.maximum(closes, 1e-10))
    rets = np.diff(log_prices, prepend=log_prices[0])
    rets[0] = 0.0

    # Rolling Fourier decomposition
    min_window = period * 2
    for i in range(min_window, n):
        start = max(0, i - period * 3)
        window = rets[start:i + 1]
        wn = len(window)

        # Detrend
        trend = np.linspace(window[0], window[-1], wn)
        detrended = window - trend

        # FFT
        fft_vals = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(wn)

        # Keep only components near the seasonal frequency
        target_freq = 1.0 / period
        bandwidth = target_freq * 0.3  # 30% bandwidth

        mask = (np.abs(freqs - target_freq) > bandwidth)
        fft_filtered = fft_vals.copy()
        fft_filtered[mask] = 0.0

        # Reconstruct seasonal
        seasonal = np.fft.irfft(fft_filtered, n=wn)

        seasonal_component[i] = seasonal[-1]

        total_var = np.var(detrended)
        seasonal_var = np.var(seasonal)
        if total_var > 0:
            strength_ratio[i] = np.clip(seasonal_var / total_var, 0.0, 1.0)
        else:
            strength_ratio[i] = 0.0

    return seasonal_component, strength_ratio
