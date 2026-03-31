import numpy as np


def hmm_regime(closes, n_states=3, period=252):
    """Simple Hidden Markov Model regime detection using Gaussian emissions.

    Implements a basic EM (Baum-Welch) algorithm with diagonal Gaussian
    emissions.  No external HMM library is used.  The model is retrained
    every ``period // 4`` bars on the trailing window.

    Parameters
    ----------
    closes : 1-D array of close prices.
    n_states : number of hidden states.
    period : rolling window for training.

    Returns
    -------
    state_labels : (N,) MAP state label at each bar (0, 1, ..., n_states-1).
    state_probs : (N, n_states) posterior probability of each state.
    transition_matrix_diag : (N,) diagonal of transition matrix for current state
        (self-transition probability = regime persistence).
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    K = n_states
    state_labels = np.full(n, np.nan, dtype=np.float64)
    state_probs = np.full((n, K), np.nan, dtype=np.float64)
    transition_matrix_diag = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return state_labels, state_probs, transition_matrix_diag

    # use log returns as observations
    log_ret = np.full(n, np.nan, dtype=np.float64)
    log_ret[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-10))

    def _gaussian_pdf(x, mu, var):
        """Vectorised Gaussian pdf."""
        return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)

    def _forward(obs, pi, A, means, varis):
        """Forward algorithm, returns alpha (T, K) and log-likelihood."""
        T = len(obs)
        alpha = np.zeros((T, K), dtype=np.float64)
        scale = np.zeros(T, dtype=np.float64)

        # t=0
        for j in range(K):
            alpha[0, j] = pi[j] * _gaussian_pdf(obs[0], means[j], varis[j])
        scale[0] = alpha[0].sum()
        if scale[0] > 0:
            alpha[0] /= scale[0]

        for t in range(1, T):
            for j in range(K):
                alpha[t, j] = np.dot(alpha[t - 1], A[:, j]) * _gaussian_pdf(obs[t], means[j], varis[j])
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]

        ll = np.sum(np.log(np.maximum(scale, 1e-300)))
        return alpha, scale, ll

    def _backward(obs, A, means, varis, scale):
        """Backward algorithm."""
        T = len(obs)
        beta = np.zeros((T, K), dtype=np.float64)
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            for j in range(K):
                s = 0.0
                for k in range(K):
                    s += A[j, k] * _gaussian_pdf(obs[t + 1], means[k], varis[k]) * beta[t + 1, k]
                beta[t, j] = s
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]
        return beta

    def _baum_welch(obs, n_iter=15):
        """EM for HMM with Gaussian emissions."""
        T = len(obs)
        # initialise with k-means-like split
        sorted_obs = np.sort(obs)
        means = np.array([sorted_obs[int(T * (k + 0.5) / K)] for k in range(K)])
        varis = np.full(K, np.var(obs) + 1e-10)
        pi = np.ones(K) / K
        A = np.full((K, K), 1.0 / K)
        # bias toward self-transition
        for k in range(K):
            A[k, k] = 0.8
            off = 0.2 / max(K - 1, 1)
            for j in range(K):
                if j != k:
                    A[k, j] = off

        for _ in range(n_iter):
            alpha, scale, ll = _forward(obs, pi, A, means, varis)
            beta = _backward(obs, A, means, varis, scale)

            # gamma
            gamma = alpha * beta
            g_sum = gamma.sum(axis=1, keepdims=True)
            g_sum[g_sum < 1e-300] = 1e-300
            gamma /= g_sum

            # xi
            xi = np.zeros((K, K), dtype=np.float64)
            for t in range(T - 1):
                denom = 0.0
                xi_t = np.zeros((K, K), dtype=np.float64)
                for i_ in range(K):
                    for j_ in range(K):
                        v = alpha[t, i_] * A[i_, j_] * _gaussian_pdf(obs[t + 1], means[j_], varis[j_]) * beta[t + 1, j_]
                        xi_t[i_, j_] = v
                        denom += v
                if denom > 1e-300:
                    xi += xi_t / denom

            # update
            pi = gamma[0] / gamma[0].sum()

            for k in range(K):
                g_k = gamma[:, k]
                g_sum_k = g_k.sum()
                if g_sum_k < 1e-10:
                    continue
                means[k] = np.dot(g_k, obs) / g_sum_k
                diff = obs - means[k]
                varis[k] = np.dot(g_k, diff ** 2) / g_sum_k + 1e-10

            for i_ in range(K):
                row_sum = xi[i_].sum()
                if row_sum > 1e-10:
                    A[i_] = xi[i_] / row_sum

        return pi, A, means, varis

    retrain_interval = max(1, period // 4)
    cur_pi = None
    cur_A = None
    cur_means = None
    cur_varis = None

    for i in range(period - 1, n):
        need_retrain = (cur_pi is None) or ((i - (period - 1)) % retrain_interval == 0)

        if need_retrain:
            win = log_ret[i - period + 1: i + 1]
            valid_mask = ~np.isnan(win)
            win_clean = win[valid_mask]
            if len(win_clean) < K * 3:
                continue
            try:
                cur_pi, cur_A, cur_means, cur_varis = _baum_welch(win_clean)
            except Exception:
                continue

        if cur_pi is None:
            continue

        x = log_ret[i]
        if np.isnan(x):
            continue

        # compute posterior for current observation using last alpha
        posterior = np.zeros(K, dtype=np.float64)
        for k in range(K):
            posterior[k] = cur_pi[k] * _gaussian_pdf(x, cur_means[k], cur_varis[k])
        p_sum = posterior.sum()
        if p_sum > 1e-300:
            posterior /= p_sum

        best = int(np.argmax(posterior))
        state_labels[i] = best
        state_probs[i] = posterior
        transition_matrix_diag[i] = cur_A[best, best]

    return state_labels, state_probs, transition_matrix_diag
