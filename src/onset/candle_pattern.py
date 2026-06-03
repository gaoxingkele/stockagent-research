"""K2 -- dynamic 3-12 bar pattern assembler.

Assembles the per-bar K1 candle features over a recent window into:
  - a FLAT hand-engineered vector (the recent `recent` bars' geometry + their
    relative position vs the prior `prior` bars) for LightGBM (K3), and
  - a [N, W, n_geom] geometry SEQUENCE for the learned encoder (K4).

Point-in-time: K1 features at row t depend only on bars <= t (shift/rolling are
backward-only), and we gather the window ending at the anchor's trade_date, so
no information at or after the prediction horizon leaks.

CPU-only, pandas/numpy. Reuses build_anchor_sequences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.onset.candle_geometry import panel_candle_features, FEATURE_COLS
from src.models.tcn_cross_attn import build_anchor_sequences

__all__ = ["anchor_sequences", "anchor_features"]


def _panel_with_feats(panel: pd.DataFrame, prior: int) -> pd.DataFrame:
    pf = panel.copy()
    feats = panel_candle_features(panel, prior=prior)
    for c in FEATURE_COLS:
        pf[c] = feats[c].values
    return pf


def anchor_sequences(panel: pd.DataFrame, anchor_keys: pd.DataFrame,
                     window: int = 12, prior: int = 9):
    """[N, window, n_geom] candle-geometry sequence ending at each anchor + mask."""
    pf = _panel_with_feats(panel, prior)
    X, mask = build_anchor_sequences(pf, anchor_keys, FEATURE_COLS, window)
    return X, mask


def anchor_features(panel: pd.DataFrame, anchor_keys: pd.DataFrame,
                    recent: int = 3, prior: int = 9):
    """Flat feature frame: the last `recent` bars' candle features (relative to
    the prior `prior` bars) flattened. Returns (DataFrame, valid_mask)."""
    X, mask = anchor_sequences(panel, anchor_keys, window=recent, prior=prior)
    # X[:, -1, :] is the anchor bar, X[:, -2, :] one bar earlier, ...
    cols = [f"{c}_lag{recent - 1 - j}" for j in range(recent) for c in FEATURE_COLS]
    flat = X.reshape(len(X), recent * len(FEATURE_COLS))
    return pd.DataFrame(flat, columns=cols), mask
