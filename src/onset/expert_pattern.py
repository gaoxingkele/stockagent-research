"""V12.31 user-oral expert pattern for bullish movement onset.

Encodes Round 1 (2026-05-29) of user interview into executable rules.
Used both:
  1. Standalone classifier baseline (rule-based, no ML)
  2. Feature engineering for LGBM/MLP/Transformer
  3. As prior knowledge in LLM agent prompts (see prompts/v12_31_expert_v1.md)

Round 1 captures:
  - bottoms_rising:        low_5d >= low_20d * 0.98 (2% tolerance)
  - above_5d_low_5pct:     close >= low_5d * 1.05
  - ma_pattern_ok:         MA5/10/20 spread < 3% AND MA5 upturn
  - volume_boost (BONUS):  vol_5d_mean > vol_20d_mean * 1.2 (additive, not required)

Bearish onset is asymmetric (avoidance signal only, never short).
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def bullish_onset_rules(
    df: pd.DataFrame,
    *,
    group_col: str = "ts_code",
    bottoms_tolerance: float = 0.02,
    above_low_pct: float = 0.05,
    ma_spread_max: float = 0.03,
    vol_boost_ratio: float = 1.2,
    ma_short_window: int = 5,
    ma_mid_window: int = 10,
    ma_long_window: int = 20,
    low_short_window: int = 5,
    low_long_window: int = 20,
    vol_long_window: int = 20,
) -> pd.DataFrame:
    """Vectorized computation of bullish onset rule components.

    Args:
        df: panel sorted by (group_col, date), must contain ['close','low','vol']
        bottoms_tolerance: how much low_5d may dip below low_20d (default 2%)
        above_low_pct: required gap above 5-day low (default 5%)
        ma_spread_max: max relative spread of MA5/MA10/MA20 (default 3%)
        vol_boost_ratio: bonus threshold for vol_5d/vol_20d (default 1.2)

    Returns:
        DataFrame with columns:
          low_5d, low_20d, bottoms_rising,
          above_5d_low_5pct,
          ma5, ma10, ma20, ma_spread, ma5_upward, ma_pattern_ok,
          volume_boost,
          is_bullish_onset (bool, AND of 3 required),
          onset_score (int 0-4, components summed including volume bonus)
    """
    g_low = df.groupby(group_col)["low"]
    g_close = df.groupby(group_col)["close"]
    g_vol = df.groupby(group_col)["vol"] if "vol" in df.columns else None

    out = pd.DataFrame(index=df.index)

    # Rolling lows
    low_5d = g_low.rolling(low_short_window, min_periods=low_short_window).min().reset_index(level=0, drop=True)
    low_20d = g_low.rolling(low_long_window, min_periods=low_long_window).min().reset_index(level=0, drop=True)
    out["low_5d"] = low_5d
    out["low_20d"] = low_20d

    # Rule 1: bottoms rising (with tolerance)
    out["bottoms_rising"] = low_5d >= low_20d * (1.0 - bottoms_tolerance)

    # Rule 2: above 5-day low by ≥ 5%
    out["above_5d_low_5pct"] = df["close"] >= low_5d * (1.0 + above_low_pct)

    # Rule 3a: MA convergence
    ma5 = g_close.rolling(ma_short_window, min_periods=ma_short_window).mean().reset_index(level=0, drop=True)
    ma10 = g_close.rolling(ma_mid_window, min_periods=ma_mid_window).mean().reset_index(level=0, drop=True)
    ma20 = g_close.rolling(ma_long_window, min_periods=ma_long_window).mean().reset_index(level=0, drop=True)
    out["ma5"] = ma5
    out["ma10"] = ma10
    out["ma20"] = ma20

    ma_max = pd.concat([ma5, ma10, ma20], axis=1).max(axis=1)
    ma_min = pd.concat([ma5, ma10, ma20], axis=1).min(axis=1)
    ma_spread = (ma_max - ma_min) / df["close"].replace(0, np.nan)
    out["ma_spread"] = ma_spread

    # Rule 3b: MA5 upturn (compare recent 2-day MA5 mean vs prior 5d-to-2d MA5 mean)
    ma5_recent = ma5.groupby(df[group_col]).rolling(2).mean().reset_index(level=0, drop=True)
    ma5_prior = ma5.groupby(df[group_col]).shift(2).rolling(3).mean().reset_index(level=0, drop=True)
    out["ma5_upward"] = ma5_recent > ma5_prior

    out["ma_pattern_ok"] = (ma_spread < ma_spread_max) & out["ma5_upward"].fillna(False)

    # Rule 4 (BONUS): volume boost
    if g_vol is not None:
        vol_5d = g_vol.rolling(low_short_window, min_periods=low_short_window).mean().reset_index(level=0, drop=True)
        vol_20d = g_vol.rolling(vol_long_window, min_periods=vol_long_window).mean().reset_index(level=0, drop=True)
        out["volume_boost"] = vol_5d > vol_20d * vol_boost_ratio
    else:
        out["volume_boost"] = False

    # Final: 3 required + optional bonus
    out["is_bullish_onset"] = (
        out["bottoms_rising"].fillna(False)
        & out["above_5d_low_5pct"].fillna(False)
        & out["ma_pattern_ok"].fillna(False)
    )

    # Score: 0-4 (each required = 1, volume bonus = 0.5 — but we round to int for simplicity)
    out["onset_score"] = (
        out["bottoms_rising"].astype("int8")
        + out["above_5d_low_5pct"].astype("int8")
        + out["ma_pattern_ok"].astype("int8")
        + out["volume_boost"].fillna(False).astype("int8")
    )

    return out


def bearish_onset_avoidance_rules(
    df: pd.DataFrame,
    *,
    group_col: str = "ts_code",
) -> pd.DataFrame:
    """Bearish onset = avoidance signal (NEVER for shorting in A-shares).

    Symmetric to bullish but with opposite sign on each rule. Used only to
    exclude candidates from buy-side selection (matches v3c production design).
    """
    # Mirror bullish onset with opposite direction
    g_high = df.groupby(group_col)["high"]
    g_close = df.groupby(group_col)["close"]

    high_5d = g_high.rolling(5, min_periods=5).max().reset_index(level=0, drop=True)
    high_20d = g_high.rolling(20, min_periods=20).max().reset_index(level=0, drop=True)

    tops_falling = high_5d <= high_20d * 1.02  # tops descending
    below_5d_high = df["close"] <= high_5d * 0.95  # already 5% below 5d high

    ma5 = g_close.rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    ma10 = g_close.rolling(10, min_periods=10).mean().reset_index(level=0, drop=True)
    ma20 = g_close.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    ma_max = pd.concat([ma5, ma10, ma20], axis=1).max(axis=1)
    ma_min = pd.concat([ma5, ma10, ma20], axis=1).min(axis=1)
    ma_spread = (ma_max - ma_min) / df["close"].replace(0, np.nan)

    ma5_recent = ma5.groupby(df[group_col]).rolling(2).mean().reset_index(level=0, drop=True)
    ma5_prior = ma5.groupby(df[group_col]).shift(2).rolling(3).mean().reset_index(level=0, drop=True)
    ma5_downward = ma5_recent < ma5_prior
    ma_pattern_bearish = (ma_spread < 0.03) & ma5_downward.fillna(False)

    out = pd.DataFrame(index=df.index)
    out["is_bearish_onset_avoid"] = (
        tops_falling.fillna(False)
        & below_5d_high.fillna(False)
        & ma_pattern_bearish.fillna(False)
    )
    return out
