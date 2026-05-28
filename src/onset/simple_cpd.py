"""Simple online change-point detector for adaptive backward windows.

Lightweight surrogate for full BOCPD (Adams & MacKay 2007), used to validate
the *adaptive vs fixed window* hypothesis (paper §2.6 Definition 4) before
committing to the full HSSM implementation.

Mechanism (per stock):
  At time t, declare a change-point if |r_t| > k * rolling_std(r[t-W:t]).
  Then adaptive_window(t) = t - last_changepoint, and
       adaptive_cumret(t) = cumprod of returns since last CP.

This captures the spirit of BOCPD's "run length" without the full Bayesian
machinery. If this simple version fails on Gate 1, full BOCPD is unlikely
to save the main paper claim either.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from numba import njit  # type: ignore
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):  # noqa: D103
        if args and callable(args[0]):
            return args[0]
        def deco(f):
            return f
        return deco


@njit(cache=True)
def _adaptive_cumret_one_stock(
    returns: np.ndarray, lookback: int, sigma_mult: float,
) -> np.ndarray:
    """Compute since-last-changepoint cumulative return for one stock."""
    n = returns.shape[0]
    out = np.zeros(n, dtype=np.float64)
    last_cp = 0
    for t in range(n):
        # Detect change-point at t (need history >= lookback to estimate vol)
        if t > lookback:
            mu = 0.0
            for i in range(t - lookback, t):
                mu += returns[i]
            mu /= lookback
            var = 0.0
            for i in range(t - lookback, t):
                d = returns[i] - mu
                var += d * d
            var /= max(lookback - 1, 1)
            sd = np.sqrt(var) if var > 0 else 1e-8
            if abs(returns[t] - mu) > sigma_mult * sd:
                last_cp = t
        # Cumulative simple return since last_cp INCLUSIVE.
        # The CP-day return is the onset's defining move and must be captured.
        cum = 0.0
        for i in range(last_cp, t + 1):
            cum = (1.0 + cum) * (1.0 + returns[i]) - 1.0
        out[t] = cum
    return out


def adaptive_cumret(
    df: pd.DataFrame,
    return_col: str = "_ret_1",
    group_col: str = "ts_code",
    lookback: int = 20,
    sigma_mult: float = 2.0,
) -> pd.Series:
    """Vectorized over stocks: compute adaptive (since-last-CP) cumulative return.

    Args:
        df: panel sorted by (group_col, date)
        return_col: daily simple return column
        lookback: rolling window for volatility estimation
        sigma_mult: change-point threshold in sigma units
    """
    logger.info(
        f"Adaptive CPD: lookback={lookback}, sigma_mult={sigma_mult} "
        f"(numba {'on' if _HAS_NUMBA else 'OFF — slow fallback'})"
    )
    out = np.zeros(len(df), dtype=np.float64)
    rets = df[return_col].values
    # Replace NaN with 0 inside numba scope (numba can't handle NaN in arithmetic cleanly)
    rets = np.where(np.isnan(rets), 0.0, rets)
    for code, idx in df.groupby(group_col, sort=False).indices.items():
        arr = rets[idx]
        out[idx] = _adaptive_cumret_one_stock(arr, lookback, sigma_mult)
    return pd.Series(out, index=df.index)
