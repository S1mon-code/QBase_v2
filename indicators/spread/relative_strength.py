"""Relative strength of an asset versus a benchmark.

Computes the RS line (asset / benchmark), its momentum, and flags
new highs in the RS line.
"""

import numpy as np


def relative_strength(
    asset_closes: np.ndarray,
    benchmark_closes: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative strength: asset / benchmark, normalized.

    Parameters
    ----------
    asset_closes     : close prices of the asset.
    benchmark_closes : close prices of the benchmark.
    period           : lookback for RS momentum and new-high detection.

    Returns
    -------
    (rs_line, rs_momentum, rs_new_high)
        rs_line     – asset / benchmark ratio (normalized to start at 1).
        rs_momentum – rate of change of the RS line over *period* bars.
        rs_new_high – boolean array; True where RS line makes a new
                      rolling high within the lookback window.
    """
    n = len(asset_closes)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, np.array([], dtype=bool)

    safe_bench = np.where(benchmark_closes == 0, np.nan, benchmark_closes)
    rs_raw = asset_closes / safe_bench

    # Normalize to start at 1
    first_valid = np.nan
    for v in rs_raw:
        if not np.isnan(v) and v != 0:
            first_valid = v
            break
    if np.isnan(first_valid):
        rs_line = rs_raw.copy()
    else:
        rs_line = rs_raw / first_valid

    rs_momentum = np.full(n, np.nan)
    rs_new_high = np.zeros(n, dtype=bool)

    for i in range(period, n):
        prev = rs_line[i - period]
        if not np.isnan(prev) and prev != 0:
            rs_momentum[i] = (rs_line[i] / prev - 1.0) * 100.0

    for i in range(period - 1, n):
        window = rs_line[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0 and not np.isnan(rs_line[i]):
            rs_new_high[i] = rs_line[i] >= np.max(valid)

    return rs_line, rs_momentum, rs_new_high
