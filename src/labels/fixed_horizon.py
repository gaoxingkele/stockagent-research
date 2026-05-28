"""L1: Fixed-Horizon (FH) label - the naive baseline.

y_t = sign(P_{t+h}/P_t - 1)  with optional thresholds for three-class version.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def fixed_horizon_label(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold_up: float = 0.03,
    threshold_down: float | None = None,
    price_col: str = "close",
    group_col: str = "ts_code",
    date_col: str = "trade_date",
) -> pd.Series:
    """Compute fixed-horizon three-class label.

    Args:
        df: panel data with ts_code / trade_date / close (must be pre-sorted)
        horizon: forward window in trading days
        threshold_up: upper threshold for class +1
        threshold_down: lower threshold for class -1 (default: -threshold_up)
        price_col: price column to use for return
        group_col / date_col: panel keys

    Returns:
        pd.Series of {-1, 0, +1} aligned with df index
    """
    if threshold_down is None:
        threshold_down = -threshold_up

    g = df.groupby(group_col)[price_col]
    fwd = g.shift(-horizon) / df[price_col] - 1.0

    y = pd.Series(0, index=df.index, dtype="int8")
    y.loc[fwd >= threshold_up] = 1
    y.loc[fwd <= threshold_down] = -1
    y.loc[fwd.isna()] = -127  # sentinel for missing
    return y
