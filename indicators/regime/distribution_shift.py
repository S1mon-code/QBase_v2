import numpy as np


def kl_divergence_shift(
    data: np.ndarray,
    period: int = 60,
    reference_period: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """KL divergence between recent and historical return distribution.

    Compares the distribution of returns over the recent `period` bars
    against the prior `reference_period` bars using discretized KL divergence.

    Returns (kl_div, is_shifted). High KL = distribution has changed.
    is_shifted = 1 when KL exceeds a natural threshold (> 0.5 nats).
    """
    n = len(data)
    kl_div = np.full(n, np.nan, dtype=np.float64)
    is_shifted = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return kl_div, is_shifted

    safe = np.maximum(data, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    total_needed = reference_period + period
    if len(log_ret) < total_needed:
        return kl_div, is_shifted

    n_bins = 20
    eps = 1e-10

    for i in range(total_needed, len(log_ret) + 1):
        recent = log_ret[i - period : i]
        reference = log_ret[i - total_needed : i - period]

        # Shared bin edges from combined data
        combined = np.concatenate([reference, recent])
        lo = np.min(combined)
        hi = np.max(combined)

        if hi - lo < 1e-14:
            kl_div[i] = 0.0
            is_shifted[i] = 0.0
            continue

        edges = np.linspace(lo, hi, n_bins + 1)

        # Histograms as probability distributions
        p_ref, _ = np.histogram(reference, bins=edges)
        p_rec, _ = np.histogram(recent, bins=edges)

        # Normalize with Laplace smoothing
        p_ref = (p_ref + eps) / (len(reference) + eps * n_bins)
        p_rec = (p_rec + eps) / (len(recent) + eps * n_bins)

        # KL(recent || reference)
        kl = np.sum(p_rec * np.log(p_rec / p_ref))
        kl = max(0.0, kl)

        kl_div[i] = kl
        is_shifted[i] = 1.0 if kl > 0.5 else 0.0

    return kl_div, is_shifted
