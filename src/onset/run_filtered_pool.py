"""FILT2 -- REAL: does the extreme-filtered high-conviction pool have a
long-only edge the broad cross-section lacked?

The pool is RULE-BASED (no training): apply pool_mask over D1, then each date the
pool's long-only market-excess return = mean(pool _fwd_r5) - mean(all _fwd_r5).
Evaluated per-year (2022-2025 = automatic cross-period) net of realistic cost,
with block CI and the pool size per date. Small N is the point -- that is the
real product.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_filtered_pool
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_tcn_wf import D1, ROOT
from src.onset.extreme_filter import pool_mask
from src.onset.long_only import summarize_excess
from src.onset.ashare_cost import DEFAULT_ROUND_TRIP

OUT = ROOT / "results/production"


def pool_excess(panel: pd.DataFrame, in_pool: pd.Series, ret="_fwd_r5", date="trade_date") -> pd.Series:
    """Per-date (mean pool return) - (mean all return)."""
    df = panel[[date, ret]].copy(); df["in_pool"] = in_pool.values
    def per_date(s):
        sp = s[s["in_pool"]]
        if len(sp) == 0 or s[ret].notna().sum() < 5:
            return np.nan
        return sp[ret].mean() - s[ret].mean()
    return df.groupby(date).apply(per_date).dropna()


def _year_eval(excess: pd.Series, cost: float = DEFAULT_ROUND_TRIP) -> dict:
    return summarize_excess(excess, cost=cost, n_boot=500)


def run_real(onset_score_min: int = 2, top_pct: float = 0.2) -> dict:
    df = pd.read_parquet(D1); df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["_fwd_r5"] = df.groupby("ts_code")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
    df["_score"] = df.groupby("ts_code")["close"].transform(lambda s: s / s.shift(20) - 1.0)  # 20d momentum

    in_pool = pool_mask(df, score_col="_score", top_pct=top_pct, onset_score_min=onset_score_min)
    excess = pool_excess(df, in_pool)
    # pool size per date
    sizes = df.assign(p=in_pool.values).groupby("trade_date")["p"].sum()

    per_year = {}
    for yr in ("2022", "2023", "2024", "2025"):
        e = excess[[d for d in excess.index if d[:4] == yr]]
        if len(e) >= 5:
            per_year[yr] = _year_eval(e)
    out = {"onset_score_min": onset_score_min, "top_pct": top_pct,
           "round_trip_cost": DEFAULT_ROUND_TRIP,
           "pool_frac_overall": float(in_pool.mean()),
           "median_pool_size_per_date": float(sizes[sizes > 0].median()) if (sizes > 0).any() else 0.0,
           "n_dates_with_pool": int((sizes > 0).sum()),
           "per_year": per_year, "pooled": _year_eval(excess),
           "positive_years": sum(1 for v in per_year.values() if v.get("mean_per_period", 0) > 0)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "filtered_pool.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
