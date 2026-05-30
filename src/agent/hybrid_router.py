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


def regime_disaster_router(
    df: pd.DataFrame,
    *,
    disaster_col: str = "_mkt_is_disaster_month",
    lgbm_signal_col: str = "lgbm_pump_ratio",
    llm_signal_col: str = "raw_p_up",
) -> pd.Series:
    """F. Disaster months → LLM raw; normal → LGBM.

    Hypothesis: LGBM fails in extreme regimes (Split 2 RankIC -0.002 happened
    to coincide with V12.31 灾难月 202508?). Route to LLM during disasters.
    """
    lgbm_rank = _rankify(df[lgbm_signal_col])
    llm_rank = _rankify(df[llm_signal_col])
    is_disaster = df[disaster_col].fillna(False).astype(bool)
    return pd.Series(np.where(is_disaster, llm_rank, lgbm_rank), index=df.index)


def lgbm_llm_disagreement_router(
    df: pd.DataFrame,
    *,
    lgbm_signal_col: str = "lgbm_pump_ratio",
    llm_signal_col: str = "raw_p_up",
    disagreement_quantile: float = 0.75,
    boost: float = 0.20,
) -> pd.Series:
    """G. Boost LLM signal where LGBM and LLM disagree the most.

    Disagreement = |lgbm_rank - llm_rank|. High disagreement = LLM brings
    new information. Default to LGBM, boost candidates where LLM disagrees
    AND ranks them high.
    """
    lgbm_rank = _rankify(df[lgbm_signal_col])
    llm_rank = _rankify(df[llm_signal_col])
    disagreement = (lgbm_rank - llm_rank).abs()
    high_disagreement = disagreement >= disagreement.quantile(disagreement_quantile)
    llm_bullish = llm_rank >= 0.75
    boost_mask = high_disagreement & llm_bullish
    return lgbm_rank + boost_mask.astype(float) * boost


def market_regime_ensemble(
    df: pd.DataFrame,
    *,
    disaster_col: str = "_mkt_is_disaster_month",
    lgbm_signal_col: str = "lgbm_pump_ratio",
    llm_raw_col: str = "raw_p_up",
    llm_expert_col: str = "expert_p_up",
) -> pd.Series:
    """H. Disaster → LLM raw; otherwise LGBM with LLM-top-quartile boost
    (essentially H_E inside non-disaster regime).
    """
    lgbm_rank = _rankify(df[lgbm_signal_col])
    llm_raw_rank = _rankify(df[llm_raw_col])
    is_disaster = df[disaster_col].fillna(False).astype(bool)
    boost_mask = llm_raw_rank >= 0.75
    inside_normal = lgbm_rank + boost_mask.astype(float) * 0.15
    return pd.Series(np.where(is_disaster, llm_raw_rank, inside_normal), index=df.index)


def onset_signal_aware_router(
    df: pd.DataFrame,
    *,
    onset_col: str = "_exp_onset_score",
    lgbm_signal_col: str = "lgbm_pump_ratio",
    llm_signal_col: str = "raw_p_up",
    boost: float = 0.30,
) -> pd.Series:
    """I. LGBM main; for high-onset-score samples, boost LLM top picks.

    Different from B_stratum: B switches entire decision; I keeps LGBM rank
    and adds LLM boost only when expert pattern fires AND LLM agrees.
    """
    lgbm_rank = _rankify(df[lgbm_signal_col])
    llm_rank = _rankify(df[llm_signal_col])
    high_onset = df[onset_col] >= 3
    llm_strong = llm_rank >= 0.70
    boost_mask = high_onset & llm_strong
    return lgbm_rank + boost_mask.astype(float) * boost


# Convenience dispatcher
STRATEGIES = {
    "A_confidence": confidence_router,
    "B_stratum": stratum_router,
    "C_soft_ensemble": soft_ensemble,
    "D_avoid_expert_onset": avoid_expert_when_onset,
    "E_lgbm_floor_llm_boost": lgbm_floor_plus_llm_topk,
    "F_disaster_aware": regime_disaster_router,
    "G_disagreement_boost": lgbm_llm_disagreement_router,
    "H_market_regime_ensemble": market_regime_ensemble,
    "I_onset_aware_boost": onset_signal_aware_router,
}
