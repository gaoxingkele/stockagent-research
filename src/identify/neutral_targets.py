"""NB1 — market / sector neutral targets and their systematic components.

Raw forward return decomposes as:
    raw = systematic (market and/or sector common move) + idiosyncratic residual
Alpha lives in the idiosyncratic residual; the systematic part is beta. We
neutralise by cross-sectional demeaning -- per trading date for market, per
date x industry for sector -- using only existing columns (no external index
series needed).

CPU-only, pandas.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["market_neutral", "sector_neutral", "decompose"]


def market_neutral(df: pd.DataFrame, ret: str = "_fwd_r5", date: str = "trade_date"):
    """Return (idiosyncratic, systematic) where systematic = per-date cross-sectional
    mean (the market move that period) and idiosyncratic = raw - systematic."""
    systematic = df.groupby(date)[ret].transform("mean")
    return df[ret] - systematic, systematic


def sector_neutral(df: pd.DataFrame, ret: str = "_fwd_r5", date: str = "trade_date",
                   sector: str = "industry"):
    """Return (idiosyncratic, systematic) neutralising the per-date per-sector mean."""
    systematic = df.groupby([date, sector])[ret].transform("mean")
    return df[ret] - systematic, systematic


def decompose(df: pd.DataFrame, ret: str = "_fwd_r5", date: str = "trade_date",
              sector: str = "industry") -> pd.DataFrame:
    """Attach market/sector neutral residuals + systematic components as columns."""
    out = df.copy()
    out["mkt_neutral"], out["mkt_systematic"] = market_neutral(df, ret, date)
    if sector in df.columns:
        out["sec_neutral"], out["sec_systematic"] = sector_neutral(df, ret, date, sector)
    return out
