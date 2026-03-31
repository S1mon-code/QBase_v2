import numpy as np


def dominant_cycle(
    data: np.ndarray,
    period: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFT-based dominant cycle period detection.

    Applies FFT to rolling windows of log returns and identifies
    the frequency with maximum spectral power.

    Returns (dominant_period, spectral_power, cycle_strength).
    cycle_strength = ratio of dominant peak power to total power (0-1).
    """
    n = len(data)
    dominant_period = np.full(n, np.nan, dtype=np.float64)
    spectral_power = np.full(n, np.nan, dtype=np.float64)
    cycle_strength = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return dominant_period, spectral_power, cycle_strength

    safe = np.maximum(data, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    if len(log_ret) < period:
        return dominant_period, spectral_power, cycle_strength

    for i in range(period, len(log_ret) + 1):
        window = log_ret[i - period : i]

        # Detrend (remove linear trend)
        x = np.arange(len(window), dtype=np.float64)
        coeffs = np.polyfit(x, window, 1)
        detrended = window - np.polyval(coeffs, x)

        # Apply Hann window to reduce spectral leakage
        hann = np.hanning(len(detrended))
        windowed = detrended * hann

        # FFT
        fft_vals = np.fft.rfft(windowed)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(windowed))

        # Skip DC component (index 0)
        if len(power) < 3:
            continue

        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]

        total_power = np.sum(power_no_dc)
        if total_power < 1e-14:
            dominant_period[i] = np.nan
            spectral_power[i] = 0.0
            cycle_strength[i] = 0.0
            continue

        # Find dominant frequency
        peak_idx = np.argmax(power_no_dc)
        peak_freq = freqs_no_dc[peak_idx]
        peak_power = power_no_dc[peak_idx]

        if peak_freq > 1e-10:
            dominant_period[i] = 1.0 / peak_freq
        else:
            dominant_period[i] = float(period)

        spectral_power[i] = peak_power
        cycle_strength[i] = peak_power / total_power

    return dominant_period, spectral_power, cycle_strength
