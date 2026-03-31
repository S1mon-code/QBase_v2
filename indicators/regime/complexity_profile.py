import numpy as np


def complexity_profile(closes: np.ndarray,
                       scales: list = None) -> tuple:
    """Multi-scale complexity analysis.

    Computes sample entropy at different scales and derives a complexity
    index. High complexity = adaptive/complex regime, low = simple/predictable.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    scales : list of int, optional
        Scales for multi-scale analysis. Default [5, 10, 20, 40].

    Returns
    -------
    complexity_index : np.ndarray
        Weighted average entropy across scales, higher = more complex.
    scale_ratio : np.ndarray
        Ratio of fine-scale to coarse-scale entropy.
        >1 = complexity increases at fine scales (chaotic).
        <1 = complexity decreases at fine scales (structured).
    """
    if scales is None:
        scales = [5, 10, 20, 40]

    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    complexity_idx = np.full(n, np.nan)
    scale_rat = np.full(n, np.nan)

    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    max_scale = max(scales)
    window = max_scale * 3  # need enough data

    for i in range(window, n):
        entropies = []
        for s in scales:
            # Coarse-grain: average returns in blocks of size s
            block_start = i - s * (window // max_scale)
            if block_start < 1:
                block_start = 1
            segment = rets[block_start:i + 1]
            valid = segment[np.isfinite(segment)]
            if len(valid) < s * 2:
                entropies.append(np.nan)
                continue

            # Compute Shannon entropy via histogram
            n_bins = max(3, int(np.sqrt(len(valid))))
            counts, _ = np.histogram(valid, bins=n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            ent = -np.sum(probs * np.log2(probs))
            entropies.append(ent)

        valid_ent = [(s, e) for s, e in zip(scales, entropies)
                     if np.isfinite(e)]

        if len(valid_ent) < 2:
            continue

        # Complexity index: weighted average (finer scales get more weight)
        weights = np.array([1.0 / s for s, _ in valid_ent])
        values = np.array([e for _, e in valid_ent])
        weights /= weights.sum()
        complexity_idx[i] = np.sum(weights * values)

        # Scale ratio: finest / coarsest
        fine_ent = valid_ent[0][1]
        coarse_ent = valid_ent[-1][1]
        if coarse_ent > 1e-9:
            scale_rat[i] = fine_ent / coarse_ent

    return complexity_idx, scale_rat
