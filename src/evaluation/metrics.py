"""Evaluation metrics: RankIC, IR, Sharpe, TopK return.

These are the standard quant evaluation metrics used throughout the paper.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def cross_sectional_rank_ic(
    pred: pd.Series, target: pd.Series, dates: pd.Series
) -> pd.Series:
    """Cross-sectional Rank IC per date.

    Returns a Series indexed by date with Spearman correlation
    between prediction and target within each cross-section.
    """
    df = pd.DataFrame({"pred": pred, "target": target, "date": dates})
    out = df.groupby("date").apply(
        lambda d: spearmanr(d["pred"], d["target"], nan_policy="omit").correlation
        if len(d) >= 5
        else np.nan
    )
    return out


def information_ratio(ic_series: pd.Series) -> float:
    """IC IR = mean(IC) / std(IC)."""
    valid = ic_series.dropna()
    if len(valid) == 0 or valid.std() == 0:
        return np.nan
    return valid.mean() / valid.std()


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe (assume risk-free = 0 for simplicity)."""
    valid = returns.dropna()
    if len(valid) == 0 or valid.std() == 0:
        return np.nan
    return np.sqrt(periods_per_year) * valid.mean() / valid.std()


def topk_return(
    pred: pd.Series, target: pd.Series, dates: pd.Series, k: int = 20
) -> pd.Series:
    """Average target of top-K predictions per date."""
    df = pd.DataFrame({"pred": pred, "target": target, "date": dates})
    out = df.groupby("date").apply(
        lambda d: d.nlargest(min(k, len(d)), "pred")["target"].mean()
    )
    return out


def max_drawdown(equity_curve: pd.Series) -> float:
    """Max drawdown of a cumulative equity curve."""
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return dd.min()


def summary(
    pred: pd.Series,
    target: pd.Series,
    dates: pd.Series,
    k: int = 20,
) -> dict:
    """Compute all key metrics in one shot."""
    ic = cross_sectional_rank_ic(pred, target, dates)
    tk = topk_return(pred, target, dates, k=k)
    return {
        "rank_ic_mean": ic.mean(),
        "rank_ic_std": ic.std(),
        "rank_ic_ir": information_ratio(ic),
        "rank_ic_positive_rate": (ic > 0).mean(),
        f"top{k}_return_mean": tk.mean(),
        f"top{k}_return_sharpe": sharpe_ratio(tk),
        "n_dates": len(ic.dropna()),
    }
