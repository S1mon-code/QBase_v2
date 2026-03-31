import numpy as np


def recurrence_rate(data, period=60, threshold_pct=10):
    """Recurrence quantification analysis: how often does the system revisit
    past states?

    Builds a rolling recurrence matrix using the embedding
    (data[t], data[t-1], data[t-2]) and counts recurrence / determinism /
    laminarity within a rolling window.

    Parameters
    ----------
    data : 1-D array (e.g. closes or returns).
    period : rolling window size.
    threshold_pct : recurrence threshold as percentile of pairwise distances
        within the window.

    Returns
    -------
    rr : (N,) recurrence rate (fraction of recurrent points).
    determinism : (N,) fraction of recurrent points forming diagonal lines (>=2).
    laminarity : (N,) fraction of recurrent points forming vertical lines (>=2).
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    rr = np.full(n, np.nan, dtype=np.float64)
    determinism = np.full(n, np.nan, dtype=np.float64)
    laminarity = np.full(n, np.nan, dtype=np.float64)

    embed_dim = 3
    warmup = period + embed_dim - 1
    if n < warmup:
        return rr, determinism, laminarity

    for i in range(warmup - 1, n):
        win = data[i - period - embed_dim + 2: i + 1]
        if np.any(np.isnan(win)):
            continue

        # build delay embedding vectors
        m = len(win) - embed_dim + 1
        if m < 5:
            continue
        embedded = np.column_stack([win[j: j + m] for j in range(embed_dim)])

        # pairwise max-norm distances
        dists = np.zeros((m, m), dtype=np.float64)
        for a in range(m):
            diff = np.abs(embedded[a] - embedded)
            dists[a] = diff.max(axis=1)

        # threshold
        upper_tri = dists[np.triu_indices(m, k=1)]
        if len(upper_tri) == 0:
            continue
        eps = np.percentile(upper_tri, threshold_pct)
        if eps < 1e-15:
            eps = 1e-15

        rec_mat = (dists <= eps).astype(np.int8)
        np.fill_diagonal(rec_mat, 0)  # exclude self-matches

        total_pairs = m * (m - 1)
        rec_count = rec_mat.sum()
        rr[i] = rec_count / total_pairs if total_pairs > 0 else 0.0

        # determinism: diagonal lines of length >= 2
        diag_total = 0
        diag_in_lines = 0
        for k in range(1, m):
            diag = np.diag(rec_mat, k)
            diag_total += diag.sum()
            # count consecutive runs >= 2
            run = 0
            for v in diag:
                if v:
                    run += 1
                else:
                    if run >= 2:
                        diag_in_lines += run
                    run = 0
            if run >= 2:
                diag_in_lines += run
        # also count negative diagonals
        for k in range(1, m):
            diag = np.diag(rec_mat, -k)
            diag_total += diag.sum()
            run = 0
            for v in diag:
                if v:
                    run += 1
                else:
                    if run >= 2:
                        diag_in_lines += run
                    run = 0
            if run >= 2:
                diag_in_lines += run

        determinism[i] = diag_in_lines / diag_total if diag_total > 0 else 0.0

        # laminarity: vertical lines of length >= 2
        vert_total = rec_mat.sum()
        vert_in_lines = 0
        for col in range(m):
            run = 0
            for row in range(m):
                if rec_mat[row, col]:
                    run += 1
                else:
                    if run >= 2:
                        vert_in_lines += run
                    run = 0
            if run >= 2:
                vert_in_lines += run
        laminarity[i] = vert_in_lines / vert_total if vert_total > 0 else 0.0

    return rr, determinism, laminarity
