"""COMBO1 -- REAL: production-faithful strategy = TIMING x FILTERING x ASYMMETRY.

In-market only when the trend regime is up (TIM); hold the extreme-filtered
long-only pool (FILT); cash otherwise. Reports the Sharpe DECOMPOSITION
(buy-hold market vs +timing vs +timing+pool-selection) per-year + pooled, and a
SIGN-D1 verdict (does the combo beat buy-and-hold cross-period?).

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_production_faithful
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.train_tcn_wf import D1, ROOT
from src.onset.extreme_filter import pool_mask
from src.onset.run_filtered_pool import pool_excess
from src.onset.run_timing import trend_regime
from src.onset.timing_overlay import decompose_sharpe

OUT = ROOT / "results/production"


def combo_eval(market_ret: pd.Series, in_market: pd.Series, selection_excess: pd.Series) -> dict:
    return decompose_sharpe(market_ret, in_market, selection_excess=selection_excess)


def run_real(onset_score_min: int = 2, top_pct: float = 0.2) -> dict:
    df = pd.read_parquet(D1); df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["_fwd_r5"] = df.groupby("ts_code")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
    df["_score"] = df.groupby("ts_code")["close"].transform(lambda s: s / s.shift(20) - 1.0)

    market_ret = df.groupby("trade_date")["_fwd_r5"].mean().dropna()
    in_pool = pool_mask(df, score_col="_score", top_pct=top_pct, onset_score_min=onset_score_min)
    sel = pool_excess(df, in_pool)                       # pool selection excess per date
    trend_in = trend_regime(market_ret)

    # SAME full non-overlapping grid as TIM2 (do NOT restrict to pool dates --
    # that would re-phase the timing and create small-sample artifacts);
    # selection excess is 0 on non-pool dates.
    dates = sorted(set(market_ret.index) & set(trend_in.index))[::5]
    mr = market_ret.reindex(dates); im = trend_in.reindex(dates).fillna(True); se = sel.reindex(dates).fillna(0.0)

    per_year = {}
    for yr in ("2022", "2023", "2024", "2025"):
        idx = [d for d in dates if d[:4] == yr]
        if len(idx) >= 5:
            per_year[yr] = combo_eval(mr.reindex(idx), im.reindex(idx), se.reindex(idx))
    pooled = combo_eval(mr, im, se)

    beats = pooled.get("timed_plus_selected_sharpe", float("-inf")) > pooled.get("buy_hold_sharpe", float("inf"))
    pos_years = sum(1 for v in per_year.values()
                    if v.get("timed_plus_selected_sharpe", 0) > v.get("buy_hold_sharpe", 0))
    out = {"onset_score_min": onset_score_min, "top_pct": top_pct,
           "per_year": per_year, "pooled": pooled,
           "combo_beats_buyhold_pooled": bool(beats), "years_combo_beats_buyhold": pos_years,
           "verdict": ("combo beats buy-hold cross-period" if beats and pos_years >= 2
                       else "combo does NOT beat buy-hold (timing wash + selection negative)")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "combo.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
