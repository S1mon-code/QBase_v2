import numpy as np


def regime_duration(regime_labels, period=60):
    """Analyse regime persistence: how long does each regime typically last?

    Scans the regime label sequence to measure current run duration,
    average duration within a rolling window, and empirical transition
    probability.

    Parameters
    ----------
    regime_labels : 1-D integer array of regime labels (e.g. from k-means or HMM).
    period : rolling window for computing average duration and transition prob.

    Returns
    -------
    current_regime_duration : (N,) how many consecutive bars the current regime
        has been active.
    avg_regime_duration : (N,) rolling average regime duration (bars per regime
        episode) within the window.
    regime_transition_prob : (N,) rolling fraction of bars where a regime change
        occurs within the window.
    """
    regime_labels = np.asarray(regime_labels, dtype=np.float64)
    n = len(regime_labels)
    current_regime_duration = np.full(n, np.nan, dtype=np.float64)
    avg_regime_duration = np.full(n, np.nan, dtype=np.float64)
    regime_transition_prob = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return current_regime_duration, avg_regime_duration, regime_transition_prob

    # current run duration (cumulative, no window needed)
    run = 0
    for i in range(n):
        if np.isnan(regime_labels[i]):
            run = 0
            continue
        if i == 0 or np.isnan(regime_labels[i - 1]):
            run = 1
        elif regime_labels[i] == regime_labels[i - 1]:
            run += 1
        else:
            run = 1
        current_regime_duration[i] = run

    # rolling stats
    for i in range(period - 1, n):
        win = regime_labels[i - period + 1: i + 1]
        if np.any(np.isnan(win)):
            valid = win[~np.isnan(win)]
            if len(valid) < 3:
                continue
        else:
            valid = win

        # count transitions
        changes = np.sum(valid[1:] != valid[:-1])
        regime_transition_prob[i] = changes / (len(valid) - 1)

        # average duration: total bars / (number of regime episodes)
        n_episodes = changes + 1
        avg_regime_duration[i] = len(valid) / n_episodes

    return current_regime_duration, avg_regime_duration, regime_transition_prob
