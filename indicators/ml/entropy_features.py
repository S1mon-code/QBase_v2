import numpy as np
from scipy.signal import periodogram


def multi_entropy(
    closes: np.ndarray,
    period: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multiple entropy measures: permutation, spectral, approximate.

    Parameters
    ----------
    closes : (N,) price series.
    period : rolling window length.

    Returns
    -------
    perm_ent : (N,) permutation entropy (normalised, 0-1).
    spec_ent : (N,) spectral entropy (normalised, 0-1).
    approx_ent : (N,) approximate entropy.
    """
    n = len(closes)
    perm_ent = np.full(n, np.nan, dtype=np.float64)
    spec_ent = np.full(n, np.nan, dtype=np.float64)
    approx_ent = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return perm_ent, spec_ent, approx_ent

    safe = np.maximum(closes, 1e-12)
    log_ret = np.full(n, np.nan, dtype=np.float64)
    log_ret[1:] = np.log(safe[1:]) - np.log(safe[:-1])

    for i in range(period, n):
        window = log_ret[i - period + 1 : i + 1]
        if np.any(np.isnan(window)):
            continue

        perm_ent[i] = _permutation_entropy(window, order=3)
        spec_ent[i] = _spectral_entropy(window)
        approx_ent[i] = _approximate_entropy(window, m=2, r_mult=0.2)

    return perm_ent, spec_ent, approx_ent


def _permutation_entropy(x: np.ndarray, order: int = 3) -> float:
    """Permutation entropy normalised to [0, 1]."""
    n = len(x)
    if n < order:
        return np.nan
    from math import factorial

    n_perms = factorial(order)
    counts = np.zeros(n_perms, dtype=np.float64)

    for i in range(n - order + 1):
        pattern = tuple(np.argsort(x[i : i + order]))
        # Map pattern to index via ranking
        idx = 0
        for j, p in enumerate(pattern):
            idx = idx * (order - j) + p
            # Adjust for remaining positions
        # Simpler: hash approach
        idx = _pattern_to_index(pattern, order)
        if 0 <= idx < n_perms:
            counts[idx] += 1.0

    total = np.sum(counts)
    if total < 1:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    h = -np.sum(probs * np.log2(probs))
    h_max = np.log2(n_perms)
    return h / h_max if h_max > 0 else 0.0


def _pattern_to_index(pattern: tuple, order: int) -> int:
    """Lehmer code to convert permutation to unique index."""
    idx = 0
    from math import factorial

    for i in range(order):
        # Count elements after position i that are smaller
        smaller = sum(1 for j in range(i + 1, order) if pattern[j] < pattern[i])
        idx += smaller * factorial(order - 1 - i)
    return idx


def _spectral_entropy(x: np.ndarray) -> float:
    """Spectral entropy normalised to [0, 1]."""
    _, psd = periodogram(x)
    psd = psd[psd > 0]
    if len(psd) == 0:
        return 0.0
    psd_norm = psd / np.sum(psd)
    h = -np.sum(psd_norm * np.log2(psd_norm))
    h_max = np.log2(len(psd_norm))
    return h / h_max if h_max > 0 else 0.0


def _approximate_entropy(x: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """Approximate entropy."""
    n = len(x)
    r = r_mult * np.std(x)
    if r < 1e-12 or n < m + 1:
        return 0.0

    def _phi(m_val: int) -> float:
        templates = np.array([x[i : i + m_val] for i in range(n - m_val + 1)])
        count = np.zeros(len(templates))
        for i in range(len(templates)):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            count[i] = np.sum(dist <= r)
        count /= len(templates)
        return np.mean(np.log(count + 1e-12))

    return abs(_phi(m) - _phi(m + 1))
