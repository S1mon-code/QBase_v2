import numpy as np


def mama(
    closes: np.ndarray,
    fast_limit: float = 0.5,
    slow_limit: float = 0.05,
) -> tuple:
    """MESA Adaptive Moving Average (Ehlers).

    Uses Hilbert Transform to measure the instantaneous phase of the
    dominant cycle, then adapts the smoothing factor accordingly.

    Returns (mama, fama). MAMA crosses above FAMA = bullish.
    """
    if closes.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy()
    n = closes.size
    if n < 33:
        nan_arr = np.full(n, np.nan)
        return nan_arr, nan_arr.copy()

    # Smooth price
    smooth = np.zeros(n)
    for i in range(3, n):
        smooth[i] = (
            4.0 * closes[i] + 3.0 * closes[i - 1] + 2.0 * closes[i - 2] + closes[i - 3]
        ) / 10.0
    for i in range(3):
        smooth[i] = closes[i]

    detrender = np.zeros(n)
    q1 = np.zeros(n)
    i1 = np.zeros(n)
    ji = np.zeros(n)
    jq = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    period_arr = np.full(n, 6.0)
    smooth_period = np.full(n, 6.0)
    phase = np.zeros(n)
    mama_out = np.full(n, np.nan)
    fama_out = np.full(n, np.nan)

    mama_out[0] = closes[0]
    fama_out[0] = closes[0]

    for i in range(6, n):
        # Detrend
        detrender[i] = (
            0.0962 * smooth[i]
            + 0.5769 * smooth[max(0, i - 2)]
            - 0.5769 * smooth[max(0, i - 4)]
            - 0.0962 * smooth[max(0, i - 6)]
        ) * (0.075 * period_arr[i - 1] + 0.54)

        # In-phase and quadrature
        q1[i] = (
            0.0962 * detrender[i]
            + 0.5769 * detrender[max(0, i - 2)]
            - 0.5769 * detrender[max(0, i - 4)]
            - 0.0962 * detrender[max(0, i - 6)]
        ) * (0.075 * period_arr[i - 1] + 0.54)
        i1[i] = detrender[max(0, i - 3)]

        # Jitter
        ji[i] = (
            0.0962 * i1[i]
            + 0.5769 * i1[max(0, i - 2)]
            - 0.5769 * i1[max(0, i - 4)]
            - 0.0962 * i1[max(0, i - 6)]
        ) * (0.075 * period_arr[i - 1] + 0.54)
        jq[i] = (
            0.0962 * q1[i]
            + 0.5769 * q1[max(0, i - 2)]
            - 0.5769 * q1[max(0, i - 4)]
            - 0.0962 * q1[max(0, i - 6)]
        ) * (0.075 * period_arr[i - 1] + 0.54)

        # Phasor addition
        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]

        # Smooth
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1]

        # Homodyne
        re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]
        re[i] = 0.2 * re[i] + 0.8 * re[i - 1]
        im[i] = 0.2 * im[i] + 0.8 * im[i - 1]

        if abs(im[i]) > 1e-10 and abs(re[i]) > 1e-10:
            period_arr[i] = 2.0 * np.pi / np.arctan(im[i] / re[i])
        else:
            period_arr[i] = period_arr[i - 1]

        period_arr[i] = max(6.0, min(50.0, period_arr[i]))
        period_arr[i] = max(0.67 * period_arr[i - 1], min(1.5 * period_arr[i - 1], period_arr[i]))
        smooth_period[i] = 0.33 * period_arr[i] + 0.67 * smooth_period[i - 1]

        # Phase
        if abs(i1[i]) > 1e-10:
            phase[i] = np.degrees(np.arctan(q1[i] / i1[i]))
        else:
            phase[i] = phase[i - 1]

        delta_phase = phase[i - 1] - phase[i]
        delta_phase = max(1.0, delta_phase)

        alpha = max(slow_limit, fast_limit / delta_phase)

        prev_mama = mama_out[i - 1] if not np.isnan(mama_out[i - 1]) else closes[i]
        prev_fama = fama_out[i - 1] if not np.isnan(fama_out[i - 1]) else closes[i]

        mama_out[i] = alpha * closes[i] + (1.0 - alpha) * prev_mama
        fama_out[i] = 0.5 * alpha * mama_out[i] + (1.0 - 0.5 * alpha) * prev_fama

    # Set warmup NaN
    mama_out[:32] = np.nan
    fama_out[:32] = np.nan

    return mama_out, fama_out
