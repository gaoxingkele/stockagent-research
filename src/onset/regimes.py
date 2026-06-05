"""REG1 -- point-in-time regime state variables to condition the MI probe on.

Three regime variables, each an integer state per trade_date, ALL point-in-time
(only past bars; shifted), so conditioning on them never injects look-ahead:

  trend_state    : sign of the trailing market return  -> {0 down, 1 up}
  vol_state      : tercile of trailing market volatility -> {0 low, 1 mid, 2 high}
  disaster_state : disaster_filter is_disaster_month    -> {0 calm, 1 disaster}

`regime_states(df)` builds all three from the D1 panel and returns a per-date
DataFrame; `map_states_to_rows` broadcasts a date-indexed state onto anchor rows.

CPU-only, pandas/numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.onset.disaster_filter import (compute_daily_market_signals,
                                       compute_disaster_signals)

__all__ = ["trend_state", "vol_state", "disaster_state", "regime_states",
           "map_states_to_rows"]


def trend_state(market_ret: pd.Series, lookback: int = 4) -> pd.Series:
    """0 when the trailing `lookback`-period market return (PAST only, shifted) is
    <= 0, else 1. Point-in-time."""
    trailing = market_ret.rolling(lookback, min_periods=1).sum().shift(1)
    state = (trailing > 0).where(trailing.notna(), True)   # no-past -> default up
    return state.astype(int)


def vol_state(market_ret: pd.Series, lookback: int = 8) -> pd.Series:
    """Tercile (0/1/2) of trailing market volatility (rolling std, PAST only,
    shifted). Terciles computed over the whole sample -- this is a regime LABEL
    for conditioning, not a tradable signal, so global quantiles are fine."""
    vol = market_ret.rolling(lookback, min_periods=2).std().shift(1)
    out = pd.Series(1, index=market_ret.index, dtype=int)   # default mid
    v = vol.dropna()
    if len(v) >= 3 and v.nunique() >= 3:
        lo, hi = v.quantile([1 / 3, 2 / 3]).to_numpy()
        s = np.where(vol <= lo, 0, np.where(vol > hi, 2, 1))
        out = pd.Series(s, index=market_ret.index).fillna(1).astype(int)
    return out


def disaster_state(df_panel: pd.DataFrame) -> pd.Series:
    """0 calm / 1 disaster from disaster_filter (already point-in-time monthly)."""
    ms = compute_daily_market_signals(df_panel)
    ds = compute_disaster_signals(ms)
    return ds["is_disaster_month"].astype(bool).astype(int)


def regime_states(df_panel: pd.DataFrame, ret_col: str = "_fwd_r5") -> pd.DataFrame:
    """Per-trade_date DataFrame with columns trend/vol/disaster (integer states).

    The market return used for trend/vol is the cross-sectional mean of ret_col;
    that ret_col is forward, but trend/vol use ONLY the shifted trailing window, so
    state at date t depends only on returns realized strictly before t."""
    df = df_panel.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    market_ret = df.groupby("trade_date")[ret_col].mean().sort_index()
    out = pd.DataFrame(index=market_ret.index)
    out["trend"] = trend_state(market_ret)
    out["vol"] = vol_state(market_ret)
    dis = disaster_state(df).reindex(market_ret.index).fillna(0).astype(int)
    out["disaster"] = dis
    return out


def map_states_to_rows(states: pd.Series, dates: pd.Series) -> np.ndarray:
    """Broadcast a date-indexed integer state onto a column of trade_dates."""
    d = dates.astype(str).to_numpy()
    return states.reindex(d).fillna(states.mode().iloc[0] if len(states) else 0) \
                 .to_numpy().astype(int)
