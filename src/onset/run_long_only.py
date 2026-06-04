"""LO2 -- REAL: the decisive go/no-go. Does factors+candle still produce a
positive, cost-surviving LONG-ONLY top-K market-excess across all splits once
shorting is removed (A-shares cannot be shorted)?

Reuses the factors+candle feature build; trains per split on the market-neutral
target; for each split's test anchors (limit-up entries excluded) computes the
long-only top-K market-excess GROSS and NET of the realistic A-share cost
(COST1). Verdict per SIGN-D1.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_long_only
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.train_tcn_wf import D1, ROOT, SPLITS, select_feature_columns
from src.onset.candle_geometry import panel_candle_features, FEATURE_COLS
from src.models.tcn_cross_attn import build_anchor_sequences
from src.identify.neutral_targets import market_neutral
from src.onset.long_only import long_only_excess, summarize_excess
from src.onset.ashare_cost import enterable, DEFAULT_ROUND_TRIP
from src.onset.run_candle_lgbm import RECENT

OUT = ROOT / "results/deploy"
K_FRAC = 0.1


def eval_long_only(df: pd.DataFrame, sig: str = "sig", k_frac: float = K_FRAC,
                   cost: float = DEFAULT_ROUND_TRIP) -> dict:
    series = long_only_excess(df, sig=sig, k_frac=k_frac)
    return {"gross": summarize_excess(series, cost=0.0, n_boot=500),
            "net": summarize_excess(series, cost=cost, n_boot=500)}


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

    def candle_flat(anchors):
        X, m = build_anchor_sequences(pf, anchors[["ts_code", "trade_date"]], FEATURE_COLS, RECENT)
        return X.reshape(len(X), RECENT * len(FEATURE_COLS)), m

    rng = np.random.default_rng(42); per_split = {}; pooled = []
    for sid in (1, 2, 3):
        ts, te_, ve, tee = SPLITS[sid]
        trp = df[(df["trade_date"] >= ts) & (df["trade_date"] < te_) & df["_fwd_r5"].notna()]
        tr = trp.sample(min(15000, len(trp)), random_state=int(rng.integers(0, 1e6))).reset_index(drop=True)
        wf = pd.read_parquet(ROOT / f"data/processed/wf_samples_split{sid}.parquet")
        wf["trade_date"] = wf["trade_date"].astype(str); wf = wf[wf["_fwd_r5"].notna()].copy()
        if "pct_chg" not in wf.columns:
            wf["pct_chg"] = d1_idx.reindex(list(zip(wf["ts_code"], wf["trade_date"])))["pct_chg"].values

        Ctr, mtr = candle_flat(tr); Cte, mte = candle_flat(wf)
        tr = tr[mtr].reset_index(drop=True); wf = wf[mte].reset_index(drop=True)
        Ctr, Cte = Ctr[mtr], Cte[mte]
        tr["mkt_neutral"], _ = market_neutral(tr)
        Ftr = tr[factor_cols].fillna(0.0).values
        Fte = d1_idx.reindex(list(zip(wf["ts_code"], wf["trade_date"])))[factor_cols].fillna(0.0).values
        pred = _lgbm(np.hstack([Ftr, Ctr]), tr["mkt_neutral"].values, np.hstack([Fte, Cte]))

        ev = wf[["trade_date", "_fwd_r5", "pct_chg"]].copy(); ev["sig"] = pred
        ev = ev[enterable(ev)]                       # exclude un-enterable limit-up entries
        per_split[f"split{sid}"] = eval_long_only(ev)
        pooled.append(ev)

    allp = pd.concat(pooled, ignore_index=True)
    pooled_res = eval_long_only(allp)

    net_pos = sum(1 for k, v in per_split.items() if v["net"].get("mean_per_period", 0) > 0)
    pci = pooled_res["net"].get("mean_ci95", [0, 0])
    deployable = (pci[0] > 0) and net_pos >= 2
    out = {"k_frac": K_FRAC, "round_trip_cost": DEFAULT_ROUND_TRIP,
           "per_split": per_split, "pooled": pooled_res,
           "net_positive_splits": net_pos,
           "verdict_long_only": "PASS go (deployable候选)" if deployable
           else "NO-GO (long-only net-excess not significant / not robust)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "long_only.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    print(json.dumps(run_real(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
