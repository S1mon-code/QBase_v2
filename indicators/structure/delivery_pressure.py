import numpy as np


def delivery_pressure(oi: np.ndarray, volumes: np.ndarray,
                      datetimes: np.ndarray, period: int = 20) -> tuple:
    """Delivery month pressure proxy.

    Detects accelerating OI decline near delivery periods. When OI drops
    rapidly while volume stays high, it signals forced position liquidation
    ahead of contract expiry.

    Parameters
    ----------
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volumes.
    datetimes : np.ndarray
        numpy datetime64 array.
    period : int
        Lookback period for rate-of-change computation.

    Returns
    -------
    pressure_score : np.ndarray
        Delivery pressure score. Higher = more liquidation pressure.
        Range approximately 0 to 1.
    days_to_estimated_delivery : np.ndarray
        Rough estimate of days to next delivery window (based on
        typical futures quarterly cycle).
    """
    n = len(oi)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    pressure = np.full(n, np.nan)
    days_to_delivery = np.full(n, np.nan)

    months = (datetimes.astype('datetime64[M]')
              - datetimes.astype('datetime64[Y]')).astype(int) + 1
    days = (datetimes.astype('datetime64[D]')
            - datetimes.astype('datetime64[M]')).astype(int) + 1

    # Chinese futures typically deliver in odd months or specific patterns.
    # Use a simplified model: delivery risk rises in the last 15 days of
    # delivery months (assume monthly delivery cycle for generality).
    for i in range(n):
        # Days remaining in current month (approx)
        days_left = max(1, 22 - days[i])  # ~22 trading days per month
        days_to_delivery[i] = days_left

    # OI rate of change (negative = declining)
    oi_roc = np.full(n, np.nan)
    if n > period:
        oi_roc[period:] = (oi[period:] - oi[:-period]) / np.maximum(oi[:-period], 1.0)

    # Volume/OI ratio (high = active liquidation)
    vol_oi = np.full(n, np.nan)
    for i in range(n):
        if oi[i] > 0 and np.isfinite(volumes[i]):
            vol_oi[i] = volumes[i] / oi[i]

    # Pressure = declining OI + high turnover + near month-end
    for i in range(period, n):
        if np.isnan(oi_roc[i]) or np.isnan(vol_oi[i]):
            continue

        # OI declining component (0 if OI rising)
        oi_decline = max(0.0, -oi_roc[i])

        # Turnover component (z-scored)
        window = vol_oi[max(0, i - period * 3):i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < 5:
            continue

        vol_oi_z = (vol_oi[i] - np.mean(valid)) / max(np.std(valid), 1e-10)
        turnover_score = max(0.0, vol_oi_z) / 3.0  # normalize

        # Time proximity (closer to month-end = higher)
        time_score = max(0.0, 1.0 - days_to_delivery[i] / 22.0)

        # Combined pressure
        raw = (oi_decline * 0.4 + turnover_score * 0.3 + time_score * 0.3)
        pressure[i] = np.clip(raw, 0.0, 1.0)

    return pressure, days_to_delivery
