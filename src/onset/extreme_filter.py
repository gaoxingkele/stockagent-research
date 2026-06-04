"""FILT1 -- extreme-filter high-conviction long-only pool (the V7c iron rules).

V12.31 does not trade a broad cross-section; it filters down to a TINY
high-conviction pool. We reproduce the available hard filters:
  - onset: expert_pattern.bullish_onset_rules (onset_score >= min),
  - exclude OVERHEATED (trailing 5-day return > +8% -> trend continuation, not onset),
  - exclude ZOMBIES (flat trailing-60d close: coeff-of-variation below a floor),
  - exclude WORST-10%-INDUSTRY by trailing-60d momentum,
  - keep only the TOP-percentile by a score within each date.

All point-in-time (rolling/shift are backward-only). CPU-only, pandas/numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.onset.expert_pattern import bullish_onset_rules

__all__ = ["overheated_mask", "zombie_mask", "industry_momentum_rank", "pool_mask"]


def _trailing_return(panel, window, group="ts_code", col="close"):
    return panel.groupby(group)[col].transform(lambda s: s / s.shift(window) - 1.0)


def overheated_mask(panel, window=5, thresh=0.08):
    """True where the trailing `window`-day return exceeds `thresh` (overheated)."""
    return (_trailing_return(panel, window) > thresh).fillna(False)


def zombie_mask(panel, window=60, flat_thresh=0.03):
    """True where trailing-window close coefficient-of-variation is below the
    floor (a flat/dead MA60 -> zombie)."""
    g = panel.groupby("ts_code")["close"]
    std = g.transform(lambda s: s.rolling(window, min_periods=window).std())
    mean = g.transform(lambda s: s.rolling(window, min_periods=window).mean())
    cv = std / mean.replace(0, np.nan)
    return (cv < flat_thresh).fillna(False)


def industry_momentum_rank(panel, window=60, date="trade_date", sector="industry"):
    """Per-row pct-rank (within date) of the stock's industry trailing momentum."""
    if sector not in panel.columns:
        return pd.Series(1.0, index=panel.index)
    past = _trailing_return(panel, window)
    tmp = pd.DataFrame({"d": panel[date].values, "s": panel[sector].values, "p": past.values},
                       index=panel.index)
    ind_mom = tmp.groupby(["d", "s"])["p"].transform("mean")
    return ind_mom.groupby(tmp["d"]).rank(pct=True)


def pool_mask(panel, score_col, top_pct=0.05, onset_score_min=3,
              date="trade_date", industry_floor=0.10):
    """Boolean per-row membership of the extreme-filtered high-conviction pool."""
    # bullish_onset_rules needs a (ts_code, trade_date)-sorted input with a clean
    # RangeIndex so its internal groupby-rolling aligns; compute there, map back.
    ps = panel.sort_values(["ts_code", date]).reset_index()      # 'index' col = original label
    score = bullish_onset_rules(ps)["onset_score"].to_numpy()
    onset_score = pd.Series(score, index=ps["index"].to_numpy())
    onset = (onset_score.reindex(panel.index) >= onset_score_min).fillna(False)
    imr = industry_momentum_rank(panel).reindex(panel.index)
    keep = pd.Series(onset.values
                     & ~overheated_mask(panel).values
                     & ~zombie_mask(panel).values
                     & (imr.values >= industry_floor), index=panel.index)
    mask = pd.Series(False, index=panel.index)
    sub = panel[keep]
    if len(sub) == 0:
        return mask
    thr = sub.groupby(date)[score_col].transform(lambda s: s.quantile(1 - top_pct))
    mask.loc[sub.index[sub[score_col] >= thr]] = True
    return mask
