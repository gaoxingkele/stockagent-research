"""WS1 — turn LLM signals into labeling functions (weak supervision).

Uses the LLM as a knowledge INJECTOR rather than a predictor: each LLM signal
becomes a labeling function (LF) voting {-1, 0(abstain), +1}, to be aggregated
by the T-002 generative label model (src/onset/weak_supervision.label_model).
On leakage-free data, any downstream gain from these soft labels is cleanly
attributable to LLM knowledge, not memorized future returns.

CPU-only, numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["llm_to_lf"]


def _band_vote(x: np.ndarray, center: float, band: float) -> np.ndarray:
    v = np.zeros(len(x), dtype=int)
    v[x > center + band] = 1
    v[x < center - band] = -1
    return v


def llm_to_lf(df: pd.DataFrame, *, p_up_col: str = "raw_p_up", p_band: float = 0.1,
              ratio_col: str | None = "raw_pump_ratio", ratio_band: float = 0.2,
              score_col: str | None = "_exp_onset_score") -> np.ndarray:
    """Build an N x K labeling-function matrix (votes in {-1,0,+1}) from LLM
    signals present in ``df``. Missing columns are skipped.

    - p_up:       +1 if p_up > 0.5+band, -1 if < 0.5-band else abstain
    - pump_ratio: +1 if > 1+band,       -1 if < 1-band     else abstain
    - onset_score:+1 if >= 3,           -1 if <= 1          else abstain (2 -> 0)
    """
    lfs = []
    if p_up_col in df.columns:
        lfs.append(_band_vote(df[p_up_col].to_numpy(float), 0.5, p_band))
    if ratio_col and ratio_col in df.columns:
        lfs.append(_band_vote(df[ratio_col].to_numpy(float), 1.0, ratio_band))
    if score_col and score_col in df.columns:
        s = df[score_col].to_numpy(float)
        v = np.zeros(len(s), dtype=int)
        v[s >= 3] = 1
        v[s <= 1] = -1
        lfs.append(v)
    if not lfs:
        raise ValueError("no LLM signal columns found to build labeling functions")
    return np.stack(lfs, axis=1)
