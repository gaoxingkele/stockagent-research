"""Disaster month signal aggregator + filter.

Encodes Round 3 (2026-05-29) of user interview into executable rules.

Disaster Month = vote(Signal_A, Signal_B, Signal_C) >= 2/3 where:

  Signal A: Index (AND)
    sh_index_today < -2.0%  AND  gem_index_today < -3.0%

  Signal B: Volume (OR — any of 3)
    B1: amount_5d_mean / amount_20d_mean < 0.70
    B2: limit_down_count > 100  OR  limit_down/limit_up > 3.0
    B3: up_stock_pct < 0.30  OR  down_stock_pct > 0.70

  Signal C: Sector (inner vote >= 2/3)
    C1: industry_red_pct > 0.80
    C2: all top-5 hot concepts < 0
    C3: top-5 hot concepts avg return < -1.0%

For now we approximate index returns using market-cap-weighted means since D1
does not contain index codes. Future revisions should pull official index data.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_daily_market_signals(df_panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-day market signals from D1 panel data.

    Args:
        df_panel: must contain ['ts_code', 'trade_date', 'pct_chg', 'amount']
                  Optional: 'industry', 'total_mv' for richer signals.

    Returns:
        DataFrame indexed by trade_date with columns:
          sh_index_pct          (approx, market-cap weighted mean)
          gem_index_pct         (approx, GEM stocks mean — codes starting with 30)
          total_amount
          amount_5d_mean        rolling mean
          amount_20d_mean       rolling mean
          amount_ratio_5_20     amount_5d / amount_20d
          limit_down_count      pct_chg < -9.5
          limit_up_count        pct_chg > 9.5
          limit_ratio           limit_down / max(limit_up, 1)
          up_stock_count        pct_chg > 0
          down_stock_count      pct_chg < 0
          up_stock_pct          fraction of stocks with positive return
          industry_red_pct      fraction of industries with negative mean return
                                (None if 'industry' missing)
    """
    df = df_panel.copy()
    df["trade_date"] = df["trade_date"].astype(str)

    # GEM proxy: codes starting with 30 (创业板 stocks)
    df["_is_gem"] = df["ts_code"].str.startswith("30")

    g = df.groupby("trade_date")
    out = pd.DataFrame(index=sorted(df["trade_date"].unique()))

    # Index approximations
    if "total_mv" in df.columns:
        # Cap-weighted mean for sh_index_pct (use total market cap weights)
        def cw(x):
            w = x["total_mv"].fillna(0)
            r = x["pct_chg"].fillna(0)
            return (w * r).sum() / max(w.sum(), 1e-9)
        out["sh_index_pct"] = g.apply(cw) / 100.0
    else:
        out["sh_index_pct"] = g["pct_chg"].mean() / 100.0

    gem_df = df[df["_is_gem"]]
    if len(gem_df) > 0:
        out["gem_index_pct"] = gem_df.groupby("trade_date")["pct_chg"].mean() / 100.0
    else:
        out["gem_index_pct"] = np.nan

    # Volume
    if "amount" in df.columns:
        total_amt = g["amount"].sum()
    elif "vol" in df.columns:
        total_amt = g["vol"].sum()
    else:
        total_amt = pd.Series(np.nan, index=out.index)
    out["total_amount"] = total_amt
    out["amount_5d_mean"] = total_amt.rolling(5, min_periods=5).mean()
    out["amount_20d_mean"] = total_amt.rolling(20, min_periods=20).mean()
    out["amount_ratio_5_20"] = out["amount_5d_mean"] / out["amount_20d_mean"].replace(0, np.nan)

    # Limit up/down (A-shares ±10%; we use 9.5 as threshold to be safe)
    out["limit_down_count"] = g.apply(lambda x: (x["pct_chg"] < -9.5).sum())
    out["limit_up_count"] = g.apply(lambda x: (x["pct_chg"] > 9.5).sum())
    out["limit_ratio"] = out["limit_down_count"] / out["limit_up_count"].replace(0, 1)

    # Market breadth
    out["up_stock_count"] = g.apply(lambda x: (x["pct_chg"] > 0).sum())
    out["down_stock_count"] = g.apply(lambda x: (x["pct_chg"] < 0).sum())
    out["total_stock_count"] = g.size()
    out["up_stock_pct"] = out["up_stock_count"] / out["total_stock_count"]
    out["down_stock_pct"] = out["down_stock_count"] / out["total_stock_count"]

    # Industry breadth (if available)
    if "industry" in df.columns:
        def ind_red_pct(x):
            ind_mean = x.groupby("industry")["pct_chg"].mean()
            return (ind_mean < 0).sum() / max(len(ind_mean), 1)
        out["industry_red_pct"] = g.apply(ind_red_pct)
    else:
        out["industry_red_pct"] = np.nan

    return out


def compute_disaster_signals(
    market_signals: pd.DataFrame,
    *,
    sh_threshold: float = -0.02,
    gem_threshold: float = -0.03,
    amount_ratio_threshold: float = 0.70,
    limit_down_count_threshold: int = 100,
    limit_ratio_threshold: float = 3.0,
    up_stock_pct_threshold: float = 0.30,
    down_stock_pct_threshold: float = 0.70,
    industry_red_threshold: float = 0.80,
) -> pd.DataFrame:
    """Compute disaster month boolean signals A, B, C and final composite.

    Notes:
      - This implementation uses placeholder NaN for top5-concept-related signals
        C2 and C3 (require concept heat data, not in D1). For now C reduces to C1.
      - Future revision: integrate concept data from
        `D:/aicoding/stockagent-analysis/output/concept_db/` to add C2/C3.
    """
    m = market_signals
    out = pd.DataFrame(index=m.index)

    # Signal A: index AND
    out["signal_A_index"] = (
        (m["sh_index_pct"] < sh_threshold)
        & (m["gem_index_pct"] < gem_threshold)
    ).fillna(False)

    # Signal B: volume OR
    b1 = (m["amount_ratio_5_20"] < amount_ratio_threshold).fillna(False)
    b2 = (
        (m["limit_down_count"] > limit_down_count_threshold)
        | (m["limit_ratio"] > limit_ratio_threshold)
    ).fillna(False)
    b3 = (
        (m["up_stock_pct"] < up_stock_pct_threshold)
        | (m["down_stock_pct"] > down_stock_pct_threshold)
    ).fillna(False)
    out["signal_B_volume"] = (b1 | b2 | b3)
    out["signal_B1_volume_shrink"] = b1
    out["signal_B2_limit_down"] = b2
    out["signal_B3_breadth"] = b3

    # Signal C: sector inner vote (C2 and C3 require concept data — TODO)
    c1 = (m["industry_red_pct"] > industry_red_threshold).fillna(False)
    # Placeholder: C2 and C3 default to False until concept data is wired in
    c2 = pd.Series(False, index=m.index)
    c3 = pd.Series(False, index=m.index)
    out["signal_C1_industry"] = c1
    out["signal_C2_concepts_all_red"] = c2
    out["signal_C3_concepts_avg_red"] = c3
    out["signal_C_sector"] = (c1.astype(int) + c2.astype(int) + c3.astype(int)) >= 2

    # Outer composite vote
    outer = (
        out["signal_A_index"].astype(int)
        + out["signal_B_volume"].astype(int)
        + out["signal_C_sector"].astype(int)
    )
    out["outer_vote_count"] = outer
    out["is_disaster_month"] = outer >= 2

    return out
