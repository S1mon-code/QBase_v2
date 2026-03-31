import numpy as np


def correlation_network_score(
    returns_matrix: np.ndarray,
    period: int = 60,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a correlation network and compute node centrality.

    For each bar, computes a rolling correlation matrix of the columns in
    ``returns_matrix``, builds an adjacency graph (edge = |corr| > threshold),
    and returns the degree centrality of the first column.

    Parameters
    ----------
    returns_matrix : (N, M) array, each column is a return series.
    period : rolling window length.
    threshold : absolute correlation threshold for an edge.

    Returns
    -------
    centrality : (N,) degree centrality of column 0 (0-1 range).
    n_connections : (N,) number of edges for column 0.
    is_central : (N,) 1.0 if centrality > 0.5, else 0.0.
    """
    n, m = returns_matrix.shape
    centrality = np.full(n, np.nan, dtype=np.float64)
    n_connections = np.full(n, np.nan, dtype=np.float64)
    is_central = np.full(n, np.nan, dtype=np.float64)

    if n < period or m < 2:
        return centrality, n_connections, is_central

    for i in range(period, n):
        window = returns_matrix[i - period : i]
        if np.any(np.isnan(window)):
            continue

        # Correlation matrix
        stds = np.std(window, axis=0)
        if np.any(stds < 1e-12):
            continue
        normed = (window - np.mean(window, axis=0)) / stds
        corr = (normed.T @ normed) / period

        # Adjacency: |corr| > threshold (exclude diagonal)
        adj = np.abs(corr) > threshold
        np.fill_diagonal(adj, False)

        # Degree centrality of column 0
        deg_0 = np.sum(adj[0])
        max_deg = m - 1
        centrality[i] = deg_0 / max_deg
        n_connections[i] = float(deg_0)
        is_central[i] = 1.0 if centrality[i] > 0.5 else 0.0

    return centrality, n_connections, is_central
