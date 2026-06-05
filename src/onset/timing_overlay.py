"""TIM1 -- market-timing overlay + Sharpe decomposition (timing vs selection).

V12.31's Sharpe plausibly comes more from TIMING (sit out disaster months) than
from cross-sectional SELECTION. This decomposes total Sharpe into:
  buy_hold          : hold the equal-weight market every period,
  timed             : hold the market only when "in-market" (cash otherwise),
  timed_plus_select : timed market + the long-only selection excess when in.
The incremental Sharpe from timing and from selection is reported.

CPU-only, pandas/numpy. Reuses long_only.summarize_excess for Sharpe + CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.onset.long_only import summarize_excess, PERIODS_PER_YEAR

__all__ = ["timed_series", "decompose_sharpe"]


def timed_series(market_ret: pd.Series, in_market: pd.Series) -> pd.Series:
    """Market return when in-market, 0 (cash) when out. Indexes aligned by date."""
    m = market_ret.copy()
    return m.where(in_market.reindex(m.index).fillna(True).astype(bool), 0.0)


def _sharpe(series: pd.Series) -> float:
    s = series.dropna()
    sd = s.std()
    return float(np.sqrt(PERIODS_PER_YEAR) * s.mean() / sd) if sd > 0 else float("nan")


def decompose_sharpe(market_ret: pd.Series, in_market: pd.Series,
                     selection_excess: pd.Series | None = None,
                     timing_cost: float = 0.0) -> dict:
    """Sharpe of buy-hold vs timed vs timed+selected, with incremental attribution.

    timing_cost is charged on each regime switch (low turnover)."""
    bh = market_ret.dropna()
    timed = timed_series(market_ret, in_market)
    # charge cost on regime switches (entering/leaving the market)
    if timing_cost > 0:
        switch = in_market.reindex(timed.index).fillna(True).astype(int).diff().abs().fillna(0)
        timed = timed - timing_cost * switch
    res = {"buy_hold_sharpe": _sharpe(bh), "timed_sharpe": _sharpe(timed),
           "n_periods": int(len(bh))}
    if selection_excess is not None:
        sel = selection_excess.reindex(timed.index).fillna(0.0)
        in_mask = in_market.reindex(timed.index).fillna(True).astype(bool)
        combined = timed + sel.where(in_mask, 0.0)
        res["timed_plus_selected_sharpe"] = _sharpe(combined)
        res["incremental_selection"] = res["timed_plus_selected_sharpe"] - res["timed_sharpe"]
    res["incremental_timing"] = res["timed_sharpe"] - res["buy_hold_sharpe"]
    return res
