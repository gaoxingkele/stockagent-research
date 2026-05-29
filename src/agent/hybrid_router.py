"""Hybrid LGBM + LLM router strategies.

We have 3 prediction signals per anchor:
  - lgbm_signal      (LGBM ratio: P_up / (P_down + 0.01))
  - llm_raw_p_up     (raw LLM probability of bullish onset)
  - llm_expert_p_up  (V12.31-augmented LLM probability of bullish onset)

Hybrid strategies combine these with awareness that:
  - LGBM is strongest in clean noise (low stratum), Top10% +2.21%
  - LLM raw is strongest in ambiguous edge cases, Top10% +3.09%
  - LLM expert is worst in onset-triggered cases (high stratum -0.28%)
    so we *avoid* using LLM expert when expert pattern is already True.

Strategies implemented:

  A. confidence_router:
     If LGBM confidence (max of {p_down, p_neutral, p_up}) >= threshold,
       use LGBM ratio.
     Else, use LLM raw p_up.

  B. stratum_router:
     - If onset_score >= 3: prefer LLM raw  (edge / high case)
     - Else:                prefer LGBM     (low / clean case)

  C. soft_ensemble:
     signal = w * lgbm_rank + (1 - w) * llm_raw_rank
     where w is LGBM confidence (higher conf → trust LGBM more).
     Signals are ranked first to be on the same scale.

  D. avoid_expert_when_onset:
     Mirrors finding-3 ("expert hurts on onset triggers"):
     - If onset_score >= 3: use LLM raw (NOT expert)
     - If 1 <= onset_score < 3: use LLM expert
     - If onset_score == 0: use LGBM

  E. lgbm_floor_plus_llm_topk:
     Use LGBM as default, but boost candidates where LLM raw is in its
     top quartile but LGBM is mid (i.e., "LLM-discovered alpha").

All strategies produce a single scalar signal aligned with the input rows.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _rankify(s: pd.Series) -> pd.Series:
    """Rank-transform a signal to [0, 1] (1 = best)."""
    return s.rank(pct=True, method="average").fillna(0.5)


def confidence_router(
    df: pd.DataFrame,
    *,
    lgbm_signal_col: str = "lgbm_pump_ratio",
    llm_signal_col: str = "raw_p_up",
    lgbm_conf_col: str | None = None,
    conf_threshold: float = 0.5,
) -> pd.Series:
    """A. Use LGBM when confident, fall back to LLM otherwise.

    `conf_col` is LGBM max-prob; if not supplied, we approximate confidence
    using the LGBM ratio's extremity (rank-based).
    """
    out = pd.Series(np.nan, index=df.index)
    if lgbm_conf_col and lgbm_conf_col in df.columns:
        conf = df[lgbm_conf_col]
    else:
        # Use absolute rank-distance from median as a proxy for LGBM confidence
        ranks = _rankify(df[lgbm_signal_col])
        conf = (ranks - 0.5).abs() * 2  # 0 at median, 1 at extremes
    use_lgbm = conf >= conf_threshold
    # When using LGBM, return ranked LGBM signal; when using LLM, ranked LLM
    lgbm_rank = _rankify(df[lgbm_signal_col])
    llm_rank = _rankify(df[llm_signal_col])
    return pd.Series(np.where(use_lgbm, lgbm_rank, llm_rank), index=df.index)


def stratum_router(
    df: pd.DataFrame,
    *,
    onset_score_col: str = "_exp_onset_score",
    lgbm_signal_col: str = "lgbm_pump_ratio",
    llm_signal_col: str = "raw_p_up",
    onset_threshold: int = 3,
) -> pd.Series:
    """B. Use LLM raw on onset/edge stratum, LGBM on clean stratum."""
    lgbm_rank = _rankify(df[lgbm_signal_col])
    llm_rank = _rankify(df[llm_signal_col])
    use_llm = df[onset_score_col] >= onset_threshold
    return pd.Series(np.where(use_llm, llm_rank, lgbm_rank), index=df.index)


def soft_ensemble(
    df: pd.DataFrame,
    *,
    lgbm_signal_col: str = "lgbm_pump_ratio",
    llm_signal_col: str = "raw_p_up",
    lgbm_weight: float = 0.6,
) -> pd.Series:
    """C. Static weighted average of ranks."""
    lgbm_rank = _rankify(df[lgbm_signal_col])
    llm_rank = _rankify(df[llm_signal_col])
    return lgbm_weight * lgbm_rank + (1 - lgbm_weight) * llm_rank


def avoid_expert_when_onset(
    df: pd.DataFrame,
    *,
    onset_score_col: str = "_exp_onset_score",
    lgbm_signal_col: str = "lgbm_pump_ratio",
    llm_raw_col: str = "raw_p_up",
    llm_expert_col: str = "expert_p_up",
) -> pd.Series:
    """D. Stratified routing aware of expert-prompt failure mode (Finding 3)."""
    lgbm_rank = _rankify(df[lgbm_signal_col])
    llm_raw_rank = _rankify(df[llm_raw_col])
    llm_exp_rank = _rankify(df[llm_expert_col])
    score = df[onset_score_col]
    return pd.Series(
        np.where(score >= 3, llm_raw_rank,           # high-stratum: use raw (not expert)
                 np.where(score >= 1, llm_exp_rank,  # edge: use expert (most context)
                          lgbm_rank)),                # clean: use LGBM
        index=df.index,
    )


def lgbm_floor_plus_llm_topk(
    df: pd.DataFrame,
    *,
    lgbm_signal_col: str = "lgbm_pump_ratio",
    llm_signal_col: str = "raw_p_up",
    llm_topk_quantile: float = 0.75,
    boost: float = 0.30,
) -> pd.Series:
    """E. LGBM main + boost where LLM (raw) is in its top quartile."""
    lgbm_rank = _rankify(df[lgbm_signal_col])
    llm_rank = _rankify(df[llm_signal_col])
    boost_mask = llm_rank >= llm_topk_quantile
    return lgbm_rank + boost_mask.astype(float) * boost


# Convenience dispatcher
STRATEGIES = {
    "A_confidence": confidence_router,
    "B_stratum": stratum_router,
    "C_soft_ensemble": soft_ensemble,
    "D_avoid_expert_onset": avoid_expert_when_onset,
    "E_lgbm_floor_llm_boost": lgbm_floor_plus_llm_topk,
}
