"""RGT1 -- the simplest possible regime-gated, long-only signal.

The minimal "motif model": in the ACTIVATING regime (e.g. trend up) rank stocks
cross-sectionally by a single candle feature and go long the top-K (market-excess);
OUT of the regime, hold cash (zero excess). Net of realistic round-trip A-share
cost. If even this can't make money cross-period, no graph/point-process model
will -- so this is the honest floor for the whole motif idea.

Reuses long_only.long_only_excess (selection) + ashare_cost (cost). Point-in-time:
the regime state and the feature must both be known at decision time (SIGN-A1).

CPU-only, pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.onset.long_only import long_only_excess
from src.onset.ashare_cost import DEFAULT_ROUND_TRIP

__all__ = ["regime_gated_excess"]


def regime_gated_excess(panel: pd.DataFrame, feature: str, in_regime: pd.Series,
                        ret: str = "_fwd_r5", date: str = "trade_date",
                        k_frac: float = 0.1,
                        cost: float = DEFAULT_ROUND_TRIP) -> pd.Series:
    """Per-date long-only top-K market-excess by `feature`, ZERO on out-of-regime
    dates, net of round-trip cost on the dates we actually trade.

    in_regime : a per-date boolean Series (index = trade_date) -- True = activating
                regime (we hold the top-K), False = cash. Reindexed onto the dates.
    Returns a per-date net-excess Series over the union of tradeable dates and
    cash dates (cash dates contribute 0), so Sharpe accounts for sitting out."""
    gross = long_only_excess(panel.assign(_sig=panel[feature]),
                             sig="_sig", ret=ret, date=date, k_frac=k_frac)
    reg = in_regime.reindex(gross.index).fillna(False).astype(bool)
    # net of cost only on traded (in-regime) dates; cash dates are exactly 0
    net = (gross - cost).where(reg, 0.0)
    return net


def gated_vs_ungated(panel: pd.DataFrame, feature: str, in_regime: pd.Series,
                     ret: str = "_fwd_r5", date: str = "trade_date",
                     k_frac: float = 0.1,
                     cost: float = DEFAULT_ROUND_TRIP) -> dict:
    """Both the gated series and the always-on (ungated) net series on the same
    date grid, so the regime's marginal contribution can be isolated."""
    gross = long_only_excess(panel.assign(_sig=panel[feature]),
                             sig="_sig", ret=ret, date=date, k_frac=k_frac)
    reg = in_regime.reindex(gross.index).fillna(False).astype(bool)
    gated = (gross - cost).where(reg, 0.0)
    ungated = gross - cost
    return {"gated": gated, "ungated": ungated, "in_regime_frac": float(reg.mean())}
