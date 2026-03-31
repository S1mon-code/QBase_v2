import numpy as np


def kde_support_resistance(
    closes: np.ndarray,
    period: int = 60,
    n_levels: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kernel density estimation for price support/resistance levels.

    Estimates the price distribution over a trailing window using a
    Gaussian KDE (implemented with numpy for speed).  Support is the
    highest-density level below the current price; resistance is the
    highest-density level above.

    Parameters
    ----------
    closes : (N,) price series.
    period : rolling window for density estimation.
    n_levels : number of price grid divisions for peak detection.

    Returns
    -------
    support_level : (N,) nearest support price.
    resistance_level : (N,) nearest resistance price.
    density_at_current : (N,) density value at the current price (0-1 normalised).
    """
    n = len(closes)
    support = np.full(n, np.nan, dtype=np.float64)
    resistance = np.full(n, np.nan, dtype=np.float64)
    density_cur = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return support, resistance, density_cur

    n_grid = max(50, n_levels * 20)

    for i in range(period, n):
        window = closes[i - period : i]
        if np.any(np.isnan(window)):
            continue

        cur_price = closes[i]
        if np.isnan(cur_price):
            continue

        lo = np.min(window)
        hi = np.max(window)
        spread = hi - lo
        if spread < 1e-12:
            support[i] = lo
            resistance[i] = hi
            density_cur[i] = 1.0
            continue

        # Gaussian KDE on a grid
        grid = np.linspace(lo - spread * 0.1, hi + spread * 0.1, n_grid)
        bw = spread / max(5, period ** 0.5)  # bandwidth
        # Vectorised KDE: sum of Gaussians
        diff = grid[:, None] - window[None, :]  # (n_grid, period)
        kde_vals = np.mean(np.exp(-0.5 * (diff / bw) ** 2), axis=1) / (bw * np.sqrt(2 * np.pi))

        # Normalise
        kde_max = np.max(kde_vals)
        if kde_max < 1e-12:
            continue
        kde_norm = kde_vals / kde_max

        # Find local maxima (peaks) in the density
        peaks = []
        for j in range(1, n_grid - 1):
            if kde_vals[j] > kde_vals[j - 1] and kde_vals[j] > kde_vals[j + 1]:
                peaks.append((grid[j], kde_vals[j]))

        if len(peaks) == 0:
            # Use the global max
            peaks = [(grid[np.argmax(kde_vals)], kde_max)]

        # Sort peaks by density (strongest first)
        peaks.sort(key=lambda x: -x[1])

        # Find nearest support (below current) and resistance (above current)
        support_candidates = [p for p in peaks if p[0] <= cur_price]
        resistance_candidates = [p for p in peaks if p[0] > cur_price]

        if support_candidates:
            support[i] = support_candidates[0][0]
        else:
            support[i] = lo

        if resistance_candidates:
            resistance[i] = resistance_candidates[0][0]
        else:
            resistance[i] = hi

        # Density at current price (interpolated)
        idx = np.searchsorted(grid, cur_price)
        idx = min(idx, n_grid - 1)
        density_cur[i] = kde_norm[idx]

    return support, resistance, density_cur
