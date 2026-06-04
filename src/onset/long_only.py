"""LO1 -- long-only top-K market-excess (the tradable object on A-shares).

A-shares mostly cannot be shorted, so the realistic alpha is a LONG-ONLY top-K
basket and its excess over the equal-weight market each date:

    excess_t = mean(ret of top-K by signal on date t) - mean(ret of all on date t)

This is what you actually earn holding the basket instead of the index. Reported
with an annualized Sharpe and a date-block bootstrap CI.

CPU-only, pandas/numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252 / 5.0          # _fwd_r5 = 5-trading-day return

__all__ = ["long_only_excess", "summarize_excess"]


def long_only_excess(df: pd.DataFrame, sig: str = "sig", ret: str = "_fwd_r5",
                     date: str = "trade_date", k_frac: float = 0.1) -> pd.Series:
    """Per-date long-only top-K market-excess return series."""
    def per_date(s):
        s = s.dropna(subset=[sig, ret])
        if len(s) < 5:
            return np.nan
        k = max(1, int(len(s) * k_frac))
        top = s.sort_values(sig).iloc[-k:]
        return top[ret].mean() - s[ret].mean()
    return df.groupby(date).apply(per_date).dropna()


def summarize_excess(series: pd.Series, cost: float = 0.0, n_boot: int = 1000,
                     seed: int = 42) -> dict:
    """Mean/period, annualized Sharpe and date-block bootstrap CI of an excess
    series, optionally net of a per-period cost."""
    s = (series - cost).dropna()
    if len(s) < 2:
        return {"n_dates": int(len(s))}
    sd = s.std()
    sharpe = float(np.sqrt(PERIODS_PER_YEAR) * s.mean() / sd) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed); v = s.values; m = len(v)
    boot = [v[rng.integers(0, m, m)].mean() for _ in range(n_boot)]
    return {"mean_per_period": float(s.mean()), "annualized_sharpe": sharpe,
            "n_dates": int(m),
            "mean_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
