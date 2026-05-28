"""L2: Triple-Barrier Method label (López de Prado 2018).

For each anchor (ts_code, t), look H bars forward:
  +1 if the upper barrier (price * (1+u)) is hit first
  -1 if the lower barrier (price * (1-d)) is hit first
   0 if H expires without either barrier being hit
-127 if insufficient forward data

This is the educational/canonical implementation. For 5M+ row panels we use
a numba-accelerated kernel when available, otherwise fall back to pure NumPy
(still ~100s on 5M rows × H=20).

Reference:
  López de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 3.
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
        # Support both @njit and @njit(cache=True) usage patterns
        if args and callable(args[0]):
            return args[0]
        def deco(f):
            return f
        return deco


@njit(cache=True)
def _tbm_one_stock(prices: np.ndarray, u: float, d: float, H: int) -> np.ndarray:
    """Triple-barrier label for one stock's price series."""
    n = prices.shape[0]
    out = np.zeros(n, dtype=np.int16)  # int16 so -127 sentinel fits
    for i in range(n):
        if i + H >= n:
            out[i] = -127
            continue
        p0 = prices[i]
        upper = p0 * (1.0 + u)
        lower = p0 * (1.0 - d)
        label = 0
        for h in range(1, H + 1):
            p = prices[i + h]
            if p >= upper:
                label = 1
                break
            if p <= lower:
                label = -1
                break
        out[i] = label
    return out


def triple_barrier_label(
    df: pd.DataFrame,
    u: float = 0.08,
    d: float = 0.05,
    H: int = 20,
    price_col: str = "close",
    group_col: str = "ts_code",
) -> pd.Series:
    """Compute LdP triple-barrier label for a panel DataFrame.

    Args:
        df: panel, must be sorted by (group_col, date) ascending
        u: upper barrier (profit) as a fraction of entry price (0.08 = +8%)
        d: lower barrier (stop-loss) as a fraction of entry price (0.05 = -5%)
        H: vertical (time) barrier in trading days
        price_col / group_col: column names

    Returns:
        pd.Series of {-1, 0, +1, -127} aligned with df.index
    """
    logger.info(
        f"TBM label: u={u} d={d} H={H} "
        f"(numba {'on' if _HAS_NUMBA else 'OFF — slow fallback'})"
    )

    out = np.empty(len(df), dtype=np.int16)
    # Walk groups in order (assumes df is sorted by group_col, then date)
    for code, idx in df.groupby(group_col, sort=False).indices.items():
        prices = df[price_col].values[idx]
        labels = _tbm_one_stock(prices, u, d, H)
        out[idx] = labels

    s = pd.Series(out.astype("int16"), index=df.index)
    return s
