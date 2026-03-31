import numpy as np


def oi_flow(closes: np.ndarray, oi: np.ndarray, volumes: np.ndarray,
            period: int = 14) -> tuple:
    """Net OI flow indicator — directional OI change weighted by price.

    Similar concept to OBV but applied to open interest.  OI change
    is signed by price direction and weighted by the magnitude of the
    price move.  A rising flow line indicates positions being built
    in the direction of price.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volume (unused directly, kept for API consistency).
    period : int
        EMA smoothing period for the signal line.

    Returns
    -------
    flow : np.ndarray
        Cumulative directional OI flow.
    flow_signal : np.ndarray
        EMA of flow (signal line).
    """
    n = len(closes)
    flow = np.full(n, np.nan)
    flow_signal = np.full(n, np.nan)

    if n < 2:
        return flow, flow_signal

    flow[0] = 0.0
    for i in range(1, n):
        price_chg = closes[i] - closes[i - 1]
        oi_chg = oi[i] - oi[i - 1]

        # Sign OI change by price direction
        if price_chg > 0:
            signed_flow = abs(oi_chg)
        elif price_chg < 0:
            signed_flow = -abs(oi_chg)
        else:
            signed_flow = 0.0

        flow[i] = flow[i - 1] + signed_flow

    # EMA signal line
    alpha = 2.0 / (period + 1)
    flow_signal[1] = flow[1]
    for i in range(2, n):
        if np.isnan(flow_signal[i - 1]):
            flow_signal[i] = flow[i]
        else:
            flow_signal[i] = alpha * flow[i] + (1 - alpha) * flow_signal[i - 1]

    return flow, flow_signal
