"""ROB1 -- REAL: cross-period + liquidity + top-K robustness of the long-only
factors+candle edge.

Only the 2025 walk-forward was tested so far. Here we build 2023 and 2024
walk-forward windows from D1 (the model is a pure leakage-free tabular learner,
so earlier windows are fine), and re-evaluate the LONG-ONLY net market-excess;
plus a liquidity haircut (trade only liquid names by `amount`) and top-K
sensitivity. SIGN-D1: deployable needs the edge to HOLD across >=1 extra year.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_robustness
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.train_tcn_wf import D1, ROOT, select_feature_columns
from src.onset.candle_geometry import panel_candle_features, FEATURE_COLS
from src.models.tcn_cross_attn import build_anchor_sequences
from src.identify.neutral_targets import market_neutral
from src.onset.long_only import long_only_excess, summarize_excess
from src.onset.ashare_cost import enterable, DEFAULT_ROUND_TRIP
from src.onset.run_candle_lgbm import RECENT

OUT = ROOT / "results/deploy"
WINDOWS = {  # year -> (train_start, train_end, test_start, test_end)
    "2023": ("20220104", "20230101", "20230101", "20240101"),
    "2024": ("20220104", "20240101", "20240101", "20250101"),
}


def topk_sensitivity(df, ks=(0.05, 0.1, 0.2), cost=DEFAULT_ROUND_TRIP):
    out = {}
    for k in ks:
        s = long_only_excess(df, k_frac=k)
        out[f"k{int(k*100)}"] = summarize_excess(s, cost=cost, n_boot=400)
    return out


def liquidity_subset(df, amount_col="amount", top_frac=0.5):
    if amount_col not in df.columns:
        return df
    thr = df[amount_col].quantile(1 - top_frac)
    return df[df[amount_col] >= thr]


def _lgbm(Xtr, ytr, Xte):
    d = lgb.Dataset(Xtr, label=ytr)
    b = lgb.train(dict(objective="regression", learning_rate=0.05, num_leaves=15,
                       min_data_in_leaf=50, verbose=-1, seed=42), d, num_boost_round=120)
    return b.predict(Xte)


def run_real() -> dict:
    df = pd.read_parquet(D1); df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["_fwd_r5"] = df.groupby("ts_code")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
    factor_cols = [c for c in select_feature_columns(df) if not c.startswith("_fwd")]
    feats = panel_candle_features(df, prior=9)
    pf = df.copy()
    for c in FEATURE_COLS:
        pf[c] = feats[c].values
    d1_idx = df.set_index(["ts_code", "trade_date"])
    rng = np.random.default_rng(7)

    def candle_flat(a):
        X, m = build_anchor_sequences(pf, a[["ts_code", "trade_date"]], FEATURE_COLS, RECENT)
        return X.reshape(len(X), RECENT * len(FEATURE_COLS)), m

    def window_eval(tstart, tend, vstart, vend):
        trp = df[(df["trade_date"] >= tstart) & (df["trade_date"] < tend) & df["_fwd_r5"].notna()]
        tr = trp.sample(min(15000, len(trp)), random_state=int(rng.integers(0, 1e6))).reset_index(drop=True)
        tep = df[(df["trade_date"] >= vstart) & (df["trade_date"] < vend) & df["_fwd_r5"].notna()]
        te = tep.sample(min(6000, len(tep)), random_state=int(rng.integers(0, 1e6))).reset_index(drop=True)
        Ctr, mtr = candle_flat(tr); Cte, mte = candle_flat(te)
        tr = tr[mtr].reset_index(drop=True); te = te[mte].reset_index(drop=True)
        Ctr, Cte = Ctr[mtr], Cte[mte]
        tr["mkt_neutral"], _ = market_neutral(tr)
        Ftr = tr[factor_cols].fillna(0.0).values; Fte = te[factor_cols].fillna(0.0).values
        pred = _lgbm(np.hstack([Ftr, Ctr]), tr["mkt_neutral"].values, np.hstack([Fte, Cte]))
        ev = te[["trade_date", "_fwd_r5", "pct_chg", "amount"]].copy(); ev["sig"] = pred
        return ev[enterable(ev)]

    by_year = {}; evs = []
    for yr, (ts, te_, vs, ve) in WINDOWS.items():
        ev = window_eval(ts, te_, vs, ve)
        by_year[yr] = summarize_excess(long_only_excess(ev), cost=DEFAULT_ROUND_TRIP, n_boot=400)
        evs.append(ev)

    allev = pd.concat(evs, ignore_index=True)
    liquid = liquidity_subset(allev, top_frac=0.5)
    out = {
        "windows": WINDOWS, "round_trip_cost": DEFAULT_ROUND_TRIP,
        "per_year_net": by_year,
        "liquid_only_net": summarize_excess(long_only_excess(liquid), cost=DEFAULT_ROUND_TRIP, n_boot=400),
        "topk_sensitivity_net": topk_sensitivity(allev),
        "holds_across_years": all(v.get("mean_per_period", 0) > 0 for v in by_year.values()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "robustness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
