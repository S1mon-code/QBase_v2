import numpy as np


def attention_weights(features_matrix, target, period=60):
    """Soft attention mechanism: which features deserve attention now?

    Computes rolling correlations between each feature and the target,
    applies softmax to get attention weights, and produces a weighted
    feature signal.

    Parameters
    ----------
    features_matrix : (N, K) array of features.
    target : (N,) target signal (e.g. forward returns).
    period : rolling window for correlation computation.

    Returns
    -------
    attn_weights : (N, K) attention weights (sum to 1 across features).
    weighted_signal : (N,) attention-weighted feature combination.
    """
    features_matrix = np.asarray(features_matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n, k = features_matrix.shape
    attn_weights = np.full((n, k), np.nan, dtype=np.float64)
    weighted_signal = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return attn_weights, weighted_signal

    for i in range(period - 1, n):
        t_win = target[i - period + 1: i + 1]
        if np.any(np.isnan(t_win)):
            continue

        t_mean = t_win.mean()
        t_std = t_win.std()
        if t_std < 1e-10:
            continue

        scores = np.zeros(k, dtype=np.float64)
        current_features = np.zeros(k, dtype=np.float64)

        valid = True
        for j in range(k):
            f_win = features_matrix[i - period + 1: i + 1, j]
            if np.any(np.isnan(f_win)):
                valid = False
                break
            f_mean = f_win.mean()
            f_std = f_win.std()
            if f_std < 1e-10:
                scores[j] = 0.0
            else:
                # Pearson correlation as attention score
                scores[j] = np.dot(f_win - f_mean, t_win - t_mean) / (period * f_std * t_std)

            current_features[j] = features_matrix[i, j]

        if not valid:
            continue

        # softmax with temperature scaling
        # scale scores to prevent overflow
        scores_scaled = scores * 5.0  # temperature
        scores_scaled -= scores_scaled.max()
        exp_scores = np.exp(scores_scaled)
        weights = exp_scores / exp_scores.sum()

        attn_weights[i] = weights
        weighted_signal[i] = np.dot(weights, current_features)

    return attn_weights, weighted_signal
