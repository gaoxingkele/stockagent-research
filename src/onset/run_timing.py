"""TIM2 -- REAL: the decisive disaster-month TIMING test.

Does sitting out disaster months (per disaster_filter) improve the equal-weight
market Sharpe over buy-and-hold, across 2022-2025? If most of the Sharpe comes
from TIMING, that is where V12.31's edge lives -- not the cross-sectional pick.

Uses NON-OVERLAPPING 5-day periods (every 5th trading date) so the Sharpe is not
inflated by overlapping-window autocorrelation.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_timing
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_tcn_wf import D1, ROOT
from src.onset.disaster_filter import compute_daily_market_signals, compute_disaster_signals
from src.onset.timing_overlay import decompose_sharpe
from src.onset.long_only import summarize_excess

OUT = ROOT / "results/production"
TIMING_COST = 0.001          # ~one-way cost on each regime switch


def trend_regime(market_ret: pd.Series, lookback: int = 4, thresh: float = 0.0) -> pd.Series:
    """A standard WORKING market-timing signal: in-market when the recent trailing
    market return (PAST only via shift -> point-in-time) is above thresh. Used
    because the reproduced disaster_filter barely fires (see TIM2 finding)."""
    trailing = market_ret.rolling(lookback, min_periods=1).sum().shift(1)
    return (trailing > thresh).fillna(True)


def timing_eval(market_ret: pd.Series, is_disaster: pd.Series, cost: float = TIMING_COST) -> dict:
    in_market = ~is_disaster.reindex(market_ret.index).fillna(False).astype(bool)
    dec = decompose_sharpe(market_ret, in_market, timing_cost=cost)
    bh = summarize_excess(market_ret, n_boot=500)
    timed = market_ret.where(in_market, 0.0)
    td = summarize_excess(timed, cost=0.0, n_boot=500)
    dec["buy_hold_mean_ci95"] = bh.get("mean_ci95")
    dec["timed_mean_ci95"] = td.get("mean_ci95")
    dec["disaster_frac"] = float(is_disaster.reindex(market_ret.index).fillna(False).mean())
    dec["n_switches"] = int(in_market.astype(int).diff().abs().fillna(0).sum())
    return dec


def run_real() -> dict:
    df = pd.read_parquet(D1); df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["_fwd_r5"] = df.groupby("ts_code")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
    market_ret = df.groupby("trade_date")["_fwd_r5"].mean().dropna()

    ms = compute_daily_market_signals(df)
    ds = compute_disaster_signals(ms)
    is_disaster = ds["is_disaster_month"].astype(bool)

    # align + non-overlapping 5-day sampling
    dates = sorted(set(market_ret.index) & set(is_disaster.index))
    dates = dates[::5]
    mr = market_ret.reindex(dates); dis = is_disaster.reindex(dates)

    # two regime signals: the reproduced disaster filter (barely fires) and a
    # standard trend regime (working). Out-of-market = disaster True / trend down.
    trend_in = trend_regime(mr)
    signals = {"disaster_filter": dis, "trend_regime": ~trend_in}  # store as "is_out"

    arms = {}
    for name, is_out in signals.items():
        per_year = {}
        for yr in ("2022", "2023", "2024", "2025"):
            idx = [d for d in dates if d[:4] == yr]
            if len(idx) >= 5:
                per_year[yr] = timing_eval(mr.reindex(idx), is_out.reindex(idx))
        arms[name] = {"per_year": per_year, "pooled": timing_eval(mr, is_out),
                      "out_frac": float(is_out.reindex(dates).fillna(False).mean())}

    out = {"timing_cost": TIMING_COST, "n_periods": int(len(mr)),
           "disaster_frac_overall": float(dis.mean()), "arms": arms}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timing.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
