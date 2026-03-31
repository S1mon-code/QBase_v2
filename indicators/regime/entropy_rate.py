import numpy as np


def entropy_rate(closes: np.ndarray, period: int = 60,
                 m: int = 3) -> np.ndarray:
    """Conditional entropy (entropy rate).

    Measures the true uncertainty remaining after conditioning on the
    past ``m`` values. Lower entropy rate = more predictable.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    period : int
        Rolling window for entropy computation.
    m : int
        Embedding dimension (number of past values to condition on).

    Returns
    -------
    entropy_rate_val : np.ndarray
        Conditional entropy rate per bar (bits).
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float)

    ent_rate = np.full(n, np.nan)

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    for i in range(period + m, n):
        segment = rets[i - period + 1:i + 1]
        valid = segment[np.isfinite(segment)]
        if len(valid) < period // 2:
            continue

        # Symbolize: positive = 1, negative = 0
        symbols = (valid > 0).astype(int)

        if len(symbols) < m + 2:
            continue

        # Count (m+1)-grams and m-grams
        joint_counts = {}
        cond_counts = {}

        for j in range(len(symbols) - m):
            pattern_full = tuple(symbols[j:j + m + 1])
            pattern_cond = tuple(symbols[j:j + m])

            joint_counts[pattern_full] = joint_counts.get(pattern_full, 0) + 1
            cond_counts[pattern_cond] = cond_counts.get(pattern_cond, 0) + 1

        total_joint = sum(joint_counts.values())
        if total_joint == 0:
            continue

        # H(X_{t+1} | X_t, ..., X_{t-m+1}) = H(joint) - H(conditional)
        h_joint = 0.0
        for count in joint_counts.values():
            p = count / total_joint
            if p > 0:
                h_joint -= p * np.log2(p)

        total_cond = sum(cond_counts.values())
        h_cond = 0.0
        for count in cond_counts.values():
            p = count / total_cond
            if p > 0:
                h_cond -= p * np.log2(p)

        ent_rate[i] = h_joint - h_cond

    return ent_rate
