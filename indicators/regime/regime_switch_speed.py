import numpy as np


def switch_speed(regime_labels: np.ndarray, period: int = 60) -> tuple:
    """How fast regimes switch -- measures stability.

    Takes an array of regime labels (e.g., from vol_regime_simple or
    market_state) and computes switching statistics.

    Parameters
    ----------
    regime_labels : np.ndarray
        Integer or float regime labels (e.g., 0, 1, 2).
    period : int
        Rolling window for statistics.

    Returns
    -------
    avg_duration : np.ndarray
        Average number of bars per regime in the window.
    switch_frequency : np.ndarray
        Number of regime switches per bar in the window (0-1).
    current_duration : np.ndarray
        How many consecutive bars the current regime has lasted.
    """
    n = len(regime_labels)
    if n == 0:
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=float))

    avg_dur = np.full(n, np.nan)
    switch_freq = np.full(n, np.nan)
    cur_dur = np.full(n, np.nan)

    # Compute current duration (consecutive same label)
    for i in range(n):
        if not np.isfinite(regime_labels[i]):
            continue
        duration = 1
        j = i - 1
        while j >= 0 and np.isfinite(regime_labels[j]):
            if regime_labels[j] == regime_labels[i]:
                duration += 1
                j -= 1
            else:
                break
        cur_dur[i] = float(duration)

    # Rolling statistics
    for i in range(period, n):
        window = regime_labels[i - period:i + 1]
        valid_mask = np.isfinite(window)
        if np.sum(valid_mask) < period // 2:
            continue

        valid = window[valid_mask]

        # Count switches
        switches = 0
        for j in range(1, len(valid)):
            if valid[j] != valid[j - 1]:
                switches += 1

        switch_freq[i] = switches / (len(valid) - 1) if len(valid) > 1 else 0.0

        # Average duration: total bars / (switches + 1)
        n_regimes = switches + 1
        avg_dur[i] = len(valid) / n_regimes

    return avg_dur, switch_freq, cur_dur
