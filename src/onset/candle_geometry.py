"""K1 -- candlestick geometry + relative-position features.

Scale-free description of recent price action: per-bar candle SHAPE plus the
RELATIVE POSITION of the current bar within the prior N-bar structure. This is
the parametric, learnable form of the V12.31 / V7c expert rules (bottoms_rising,
above_5d_low, volume_boost, ...). Every feature is scale-invariant (multiplying
all prices by a constant leaves them unchanged), so the representation is robust
to price level, volatility level, and regime -- the property that broke earlier
raw-return signals.

Operates on a single stock's OHLCV history sorted by date (use
``panel_candle_features`` to map over a multi-stock panel). Point-in-time: every
feature at row t uses only bars <= t.

CPU-only, pandas/numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GEOM_COLS = ["body", "upper_wick", "lower_wick", "close_loc", "range_over_atr", "gap"]
REL_COLS = ["close_pct_prior", "breakout", "dist_low_atr", "higher_lows", "vol_ratio", "compression"]
FEATURE_COLS = GEOM_COLS + REL_COLS

__all__ = ["per_bar_geometry", "relative_position", "candle_features",
           "panel_candle_features", "GEOM_COLS", "REL_COLS", "FEATURE_COLS"]


def _atr(h, l, c, n):
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def per_bar_geometry(df: pd.DataFrame, o="open", h="high", l="low", c="close",
                     atr_n: int = 10) -> pd.DataFrame:
    O, H, L, C = df[o], df[h], df[l], df[c]
    rng = (H - L).replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["body"] = (C - O) / rng
    out["upper_wick"] = (H - np.maximum(O, C)) / rng
    out["lower_wick"] = (np.minimum(O, C) - L) / rng
    out["close_loc"] = (C - L) / rng
    out["range_over_atr"] = (H - L) / _atr(H, L, C, atr_n)
    out["gap"] = (O - C.shift(1)) / C.shift(1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def relative_position(df: pd.DataFrame, h="high", l="low", c="close", vol="vol",
                      prior: int = 9, atr_n: int = 10) -> pd.DataFrame:
    H, L, C = df[h], df[l], df[c]
    # prior-window stats EXCLUDING the current bar (shift by 1)
    prior_high = H.shift(1).rolling(prior, min_periods=prior).max()
    prior_low = L.shift(1).rolling(prior, min_periods=prior).min()
    span = (prior_high - prior_low).replace(0, np.nan)
    atr = _atr(H, L, C, atr_n)
    out = pd.DataFrame(index=df.index)
    out["close_pct_prior"] = (C - prior_low) / span
    out["breakout"] = (C > prior_high).astype(float)
    out["dist_low_atr"] = (C - prior_low) / atr
    # consecutive higher lows ending at t
    hl = (L > L.shift(1)).astype(int)
    out["higher_lows"] = hl.groupby((hl == 0).cumsum()).cumcount().astype(float)
    if vol in df.columns:
        prior_vol = df[vol].shift(1).rolling(prior, min_periods=prior).mean().replace(0, np.nan)
        out["vol_ratio"] = df[vol] / prior_vol
    else:
        out["vol_ratio"] = 1.0
    prior_rng = (H - L).shift(1).rolling(prior, min_periods=prior).mean().replace(0, np.nan)
    out["compression"] = (H - L) / prior_rng
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def candle_features(df: pd.DataFrame, prior: int = 9, atr_n: int = 10) -> pd.DataFrame:
    """Per-row candle geometry + relative-position features for one stock
    (sorted by date). Returns columns FEATURE_COLS."""
    df = df.sort_values("trade_date") if "trade_date" in df.columns else df
    geom = per_bar_geometry(df, atr_n=atr_n)
    rel = relative_position(df, prior=prior, atr_n=atr_n)
    return pd.concat([geom, rel], axis=1)[FEATURE_COLS]


def panel_candle_features(panel: pd.DataFrame, prior: int = 9, atr_n: int = 10) -> pd.DataFrame:
    """Map candle_features over a panel keyed by ts_code, returning the feature
    columns aligned to the panel index."""
    panel = panel.sort_values(["ts_code", "trade_date"])
    parts = []
    for _, g in panel.groupby("ts_code", sort=False):
        f = candle_features(g, prior=prior, atr_n=atr_n)
        f.index = g.index
        parts.append(f)
    return pd.concat(parts).reindex(panel.index)
