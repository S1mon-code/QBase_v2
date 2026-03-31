import numpy as np


def transfer_entropy(source, target, period=60, n_bins=5, lag=1):
    """Rolling transfer entropy: information flow from source to target.

    Discretises both series into ``n_bins`` equal-frequency bins inside a
    rolling window and computes TE(source -> target) using the standard
    definition based on conditional Shannon entropies.

    Parameters
    ----------
    source, target : 1-D arrays of equal length.
    period : rolling window size.
    n_bins : number of quantile bins for discretisation.
    lag : prediction lag (bars ahead).

    Returns
    -------
    te_score : (N,) array.  High TE = source helps predict target.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = len(source)
    te_score = np.full(n, np.nan, dtype=np.float64)

    if n < period + lag:
        return te_score

    def _digitise(arr, nb):
        """Bin into 0..nb-1 using quantile edges."""
        edges = np.linspace(0, 100, nb + 1)
        pcts = np.percentile(arr, edges)
        # make edges strictly increasing
        pcts[-1] += 1e-12
        return np.clip(np.searchsorted(pcts, arr, side='right') - 1, 0, nb - 1)

    def _te(s_win, t_win, nb, lg):
        """Compute TE(s -> t) for one window."""
        m = len(t_win) - lg
        if m < 10:
            return 0.0
        s_d = _digitise(s_win[:m], nb)
        t_now = _digitise(t_win[:m], nb)
        t_fut = _digitise(t_win[lg:lg + m], nb)

        joint_size = nb * nb * nb
        # indices: t_fut * nb*nb + t_now * nb + s_d
        joint_idx = t_fut * nb * nb + t_now * nb + s_d
        joint = np.bincount(joint_idx, minlength=joint_size).reshape(nb, nb, nb).astype(np.float64)

        # p(t_fut, t_now, s)
        total = joint.sum()
        if total == 0:
            return 0.0
        p_joint = joint / total

        # marginals
        p_tfut_tnow = p_joint.sum(axis=2)   # sum over s
        p_tnow_s = p_joint.sum(axis=0)       # sum over t_fut
        p_tnow = p_tnow_s.sum(axis=1)        # sum over s

        te = 0.0
        for tf in range(nb):
            for tn in range(nb):
                for s in range(nb):
                    p_j = p_joint[tf, tn, s]
                    if p_j < 1e-15:
                        continue
                    p_tf_tn = p_tfut_tnow[tf, tn]
                    p_tn_s_ = p_tnow_s[tn, s]
                    p_tn_ = p_tnow[tn]
                    if p_tf_tn < 1e-15 or p_tn_s_ < 1e-15 or p_tn_ < 1e-15:
                        continue
                    te += p_j * np.log2(p_j * p_tn_ / (p_tf_tn * p_tn_s_))
        return te

    for i in range(period + lag - 1, n):
        s_win = source[i - period - lag + 1: i + 1]
        t_win = target[i - period - lag + 1: i + 1]
        if np.any(np.isnan(s_win)) or np.any(np.isnan(t_win)):
            continue
        te_score[i] = _te(s_win, t_win, n_bins, lag)

    return te_score
