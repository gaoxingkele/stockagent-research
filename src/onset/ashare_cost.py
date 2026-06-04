"""COST1 -- realistic A-share cost / constraint model.

Replaces the crude 0.4%/period haircut with the actual A-share retail frictions:
  - commission ~0.025%/side (min fee ignored),
  - stamp duty ~0.05% on SELL only (halved in Aug 2023),
  - slippage ~0.05%/side (liquid names),
  - T+1: cannot sell same day (handled by using a forward holding return),
  - 10% daily price limit: cannot BUY a name already at limit-up.

Default round-trip ~0.2%. Parametric and documented.

CPU-only, pandas.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["round_trip_cost", "enterable", "net_excess", "DEFAULT_ROUND_TRIP"]


def round_trip_cost(commission: float = 0.00025, stamp_sell: float = 0.0005,
                    slippage: float = 0.0005) -> float:
    """Buy leg (commission + slippage) + sell leg (commission + stamp + slippage).
    Stamp duty applies to the SELL only (A-share rule)."""
    buy_cost = commission + slippage
    sell_cost = commission + stamp_sell + slippage
    return buy_cost + sell_cost


DEFAULT_ROUND_TRIP = round_trip_cost()


def enterable(df: pd.DataFrame, pct_chg: str = "pct_chg", limit: float = 9.8):
    """Boolean mask: True where a name can be BOUGHT (not already at limit-up).
    A name whose day move >= `limit` (% ) is at limit-up and cannot be entered."""
    if pct_chg not in df.columns:
        return pd.Series(True, index=df.index)
    return df[pct_chg] < limit


def net_excess(gross_excess, round_trip: float = DEFAULT_ROUND_TRIP):
    """Subtract the round-trip cost per holding period from an excess series."""
    return gross_excess - round_trip
