import numpy as np


def transition_features(
    regime_labels: np.ndarray,
    n_states: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling transition matrix features from regime labels.

    Builds a transition matrix from the last observed regime transitions
    and extracts features: self-transition probability, most likely next
    state, and regime entropy.

    Parameters
    ----------
    regime_labels : (N,) integer regime labels (0, 1, ..., n_states-1).
        NaN entries are skipped.
    n_states : number of distinct regime states.

    Returns
    -------
    self_transition_prob : (N,) probability of staying in current regime.
    most_likely_next : (N,) most likely next regime label.
    regime_entropy : (N,) entropy of transition distribution from current state.
    """
    n = len(regime_labels)
    self_trans = np.full(n, np.nan, dtype=np.float64)
    most_likely = np.full(n, np.nan, dtype=np.float64)
    entropy = np.full(n, np.nan, dtype=np.float64)

    # Build running transition counts
    trans_counts = np.zeros((n_states, n_states), dtype=np.float64)
    prev_label = -1

    for i in range(n):
        lbl = regime_labels[i]
        if np.isnan(lbl):
            continue
        lbl_int = int(lbl)
        if lbl_int < 0 or lbl_int >= n_states:
            continue

        if prev_label >= 0:
            trans_counts[prev_label, lbl_int] += 1.0

        prev_label = lbl_int

        # Compute features from current transition matrix
        row = trans_counts[lbl_int]
        row_sum = np.sum(row)
        if row_sum < 1:
            continue

        probs = row / row_sum
        self_trans[i] = probs[lbl_int]
        most_likely[i] = float(np.argmax(probs))

        # Shannon entropy of the transition distribution
        p_pos = probs[probs > 0]
        entropy[i] = -np.sum(p_pos * np.log2(p_pos))

    return self_trans, most_likely, entropy
