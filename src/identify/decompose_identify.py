"""NB2 — identify the LLM's contribution to TIMING (beta) vs SELECTION (alpha).

  - SELECTION (alpha): cross-sectional incremental contribution of the LLM over a
    tabular baseline on the market/sector-NEUTRAL target (idiosyncratic residual).
    Reuses ID2 incremental_contribution (partial rank corr, date-clustered CI).
  - TIMING (beta): does the LLM signal, aggregated per date, predict the per-date
    SYSTEMATIC move (the market/sector return that period)? Date-level rank
    correlation with a block (date) bootstrap. Few dates -> wide CI; reported.

Maps to V12-Agentic: Macro Regime agent = timing, Pattern Core = selection.

CPU-only, numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.identify.contribution import incremental_contribution, _pctrank

__all__ = ["selection_contribution", "timing_contribution"]


def selection_contribution(baseline, llm, neutral_target, dates, n_boot: int = 1000) -> dict:
    """Identified LLM contribution to idiosyncratic SELECTION (on the neutral target)."""
    return incremental_contribution(baseline, llm, neutral_target, dates, n_boot=n_boot)


def _rank_corr(a, b):
    ra, rb = _pctrank(np.asarray(a, float)), _pctrank(np.asarray(b, float))
    denom = np.linalg.norm(ra - ra.mean()) * np.linalg.norm(rb - rb.mean())
    return float(((ra - ra.mean()) * (rb - rb.mean())).sum() / denom) if denom > 0 else 0.0


def timing_contribution(llm, systematic, dates, n_boot: int = 1000, seed: int = 42,
                        ci: float = 95.0) -> dict:
    """Does the per-date LLM aggregate predict the per-date systematic move?

    Aggregates the LLM signal to the date level and rank-correlates it with the
    date-level systematic return, with a block (date) bootstrap CI.
    """
    df = pd.DataFrame({"llm": np.asarray(llm, float),
                       "syst": np.asarray(systematic, float),
                       "date": np.asarray(dates)})
    g = df.groupby("date").agg(llm=("llm", "mean"), syst=("syst", "mean")).reset_index()
    point = _rank_corr(g["llm"].values, g["syst"].values)
    rng = np.random.default_rng(seed)
    m = len(g)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)               # resample whole dates
        vals.append(_rank_corr(g["llm"].values[idx], g["syst"].values[idx]))
    lo = float(np.percentile(vals, (100 - ci) / 2))
    hi = float(np.percentile(vals, 100 - (100 - ci) / 2))
    return {"mean": float(np.mean(vals)), "point": point, "lo": lo, "hi": hi, "n_dates": int(m)}
