"""OPT1 -- REAL: walk-forward knob optimization (no in-sample tuning).

Knobs: onset_score_min in {2,3}, top_pct in {0.05,0.1,0.2}. For each walk-forward
split, choose the config maximizing the TRAIN-window net long-only pool excess,
then judge that config on the TEST window. Reports the walk-forward-selected
OUT-OF-SAMPLE performance plus the full sensitivity grid (train vs test) to show
the snooping gap. Expensive onset/filter pieces are precomputed ONCE.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_optimize_wf
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_tcn_wf import D1, ROOT
from src.onset.expert_pattern import bullish_onset_rules
from src.onset.extreme_filter import overheated_mask, zombie_mask, industry_momentum_rank
from src.onset.run_filtered_pool import pool_excess
from src.onset.long_only import summarize_excess
from src.onset.ashare_cost import DEFAULT_ROUND_TRIP

OUT = ROOT / "results/production"
SPLITS = {"2023->2024": ("2023", "2024"), "2024->2025": ("2024", "2025")}
SCORE_MINS = (2, 3)
TOP_PCTS = (0.05, 0.1, 0.2)


def select_best(train_means: dict, test_results: dict) -> dict:
    """Pick the config with the highest TRAIN net mean; return its TEST result."""
    best = max(train_means, key=lambda k: train_means[k])
    return {"selected_config": best, "train_net_mean": train_means[best],
            "test_oos": test_results[best]}


def run_real() -> dict:
    df = pd.read_parquet(D1); df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["_fwd_r5"] = df.groupby("ts_code")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
    df["_score"] = df.groupby("ts_code")["close"].transform(lambda s: s / s.shift(20) - 1.0)
    # precompute the expensive pieces ONCE (independent of the knobs)
    onset_score = bullish_onset_rules(df)["onset_score"].reindex(df.index)
    oh = overheated_mask(df).values
    zo = zombie_mask(df).values
    imr = industry_momentum_rank(df).reindex(df.index).values
    base_keep = ~oh & ~zo & (imr >= 0.10)

    def pool_excess_for(score_min, top_pct):
        keep = pd.Series(base_keep & (onset_score.values >= score_min), index=df.index)
        mask = pd.Series(False, index=df.index)
        sub = df[keep]
        if len(sub):
            thr = sub.groupby("trade_date")["_score"].transform(lambda s: s.quantile(1 - top_pct))
            mask.loc[sub.index[sub["_score"] >= thr]] = True
        return pool_excess(df, mask)

    configs = [(sm, tp) for sm in SCORE_MINS for tp in TOP_PCTS]
    excess = {c: pool_excess_for(*c) for c in configs}

    grid = {}; per_split = {}
    for name, (tr_yr, te_yr) in SPLITS.items():
        train_means, test_res = {}, {}
        for c in configs:
            e = excess[c]
            tr = e[[d for d in e.index if d[:4] == tr_yr]]
            te = e[[d for d in e.index if d[:4] == te_yr]]
            train_means[str(c)] = float((tr - DEFAULT_ROUND_TRIP).mean()) if len(tr) else -9.9
            test_res[str(c)] = summarize_excess(te, cost=DEFAULT_ROUND_TRIP, n_boot=300) if len(te) >= 5 else {}
        per_split[name] = select_best(train_means, test_res)
        grid[name] = {"train_net_mean": train_means,
                      "test_net_sharpe": {k: v.get("annualized_sharpe") for k, v in test_res.items()}}

    oos = [v["test_oos"].get("annualized_sharpe") for v in per_split.values() if v["test_oos"]]
    out = {"configs": [str(c) for c in configs], "per_split_selected": per_split, "grid": grid,
           "wf_selected_oos_sharpe": oos,
           "verdict": ("walk-forward-selected OOS positive" if oos and all(s is not None and s > 0 for s in oos)
                       else "walk-forward-selected OOS NOT positive (no config recovers an edge)")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "optimize.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
