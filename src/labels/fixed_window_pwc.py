"""L6: Fixed-Window PWC label (the v3c production baseline).

Three-class with backward-context constraint:
  +1 (bullish_onset): fwd_r5 >= up_threshold AND drawdown_5 <= dd_threshold AND past_r5 <= past_threshold
  -1 (bearish_onset): fwd_r5 <= -up_threshold AND rebound_5 <= dd_threshold AND past_r5 <= past_threshold
   0 (neutral)

Note: paper §2.6 Definition 3 — this is PWC v1 (fixed-window),
the first-order approximation of the adaptive HSSM filter.

Reference: production model r5_pump_3way_lgbm_v3c
  (commit c210d4a, alpha +1.906pp/月 vs v3b +1.508pp/月)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def fixed_window_pwc_label(
    df: pd.DataFrame,
    horizon: int = 5,
    up_threshold: float = 0.10,
    dd_threshold: float = 0.05,
    past_window: int = 5,
    past_threshold: float = 0.08,
    price_col: str = "close",
    group_col: str = "ts_code",
    date_col: str = "trade_date",
) -> pd.Series:
    """Compute PWC three-class label with backward-window contamination filter.

    Args:
        df: panel data, pre-sorted by (ts_code, trade_date)
        horizon: forward window (typically 5 days)
        up_threshold: |fwd return| threshold for onset (0.10 = 10%)
        dd_threshold: max drawdown/rebound during onset window (0.05 = 5%)
        past_window: backward window for contamination filter (5 days in v3c)
        past_threshold: cumulative past return threshold (0.08 = 8% in v3c)

    Returns:
        pd.Series of {-1, 0, +1, -127} aligned with df.index
        -127 = missing (insufficient history or future data)
    """
    p = df[price_col]
    g = df.groupby(group_col)[price_col]

    # past_r: backward cumulative return
    past_r = p / g.shift(past_window) - 1.0

    # fwd_r: forward cumulative return
    fwd_r = g.shift(-horizon) / p - 1.0

    # forward drawdown / rebound within the onset window
    fwd_dd = pd.Series(0.0, index=df.index)
    fwd_rb = pd.Series(0.0, index=df.index)
    for h in range(1, horizon + 1):
        p_h = g.shift(-h)
        # for bullish: drawdown = (min path - entry) / entry
        path_ret_h = p_h / p - 1.0
        fwd_dd = pd.concat([fwd_dd, -path_ret_h], axis=1).min(axis=1)  # most negative
        fwd_rb = pd.concat([fwd_rb, path_ret_h], axis=1).max(axis=1)   # most positive

    # Apply PWC filter + onset rules
    y = pd.Series(0, index=df.index, dtype="int8")

    past_clean = past_r.abs() <= past_threshold  # NOTE: abs to filter both directions

    bullish = (fwd_r >= up_threshold) & (fwd_dd <= dd_threshold) & past_clean
    bearish = (fwd_r <= -up_threshold) & (fwd_rb <= dd_threshold) & past_clean

    y.loc[bullish] = 1
    y.loc[bearish] = -1
    y.loc[fwd_r.isna() | past_r.isna()] = -127

    return y
