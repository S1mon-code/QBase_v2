import numpy as np


def rolling_information_ratio(returns, benchmark_returns, period=60):
    """Rolling information ratio (excess return / tracking error).

    Parameters
    ----------
    returns : 1-D array of strategy returns.
    benchmark_returns : 1-D array of benchmark returns (same length).
    period : rolling window size.

    Returns
    -------
    ir : (N,) information ratio within each rolling window.
    tracking_error : (N,) std of excess returns within window.
    excess_return : (N,) mean excess return within window.
    """
    returns = np.asarray(returns, dtype=np.float64)
    benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)
    n = len(returns)
    ir = np.full(n, np.nan, dtype=np.float64)
    tracking_error = np.full(n, np.nan, dtype=np.float64)
    excess_return = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return ir, tracking_error, excess_return

    excess = returns - benchmark_returns

    for i in range(period - 1, n):
        win = excess[i - period + 1: i + 1]
        if np.any(np.isnan(win)):
            valid = win[~np.isnan(win)]
            if len(valid) < 3:
                continue
        else:
            valid = win

        mu = valid.mean()
        te = valid.std(ddof=1)

        excess_return[i] = mu
        tracking_error[i] = te
        ir[i] = mu / te if te > 1e-10 else 0.0

    return ir, tracking_error, excess_return
