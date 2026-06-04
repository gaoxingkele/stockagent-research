"""K5 -- REAL: ablation. Does candle geometry ADD anything over the 165 smoothed
factors? Train LGBM on three feature sets (factors / factors+candle / candle)
with the same all-split market-neutral cost-aware eval; report the incremental
net long-short Sharpe of adding candle geometry.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_candle_ablation
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
from src.evaluation.onset_eval import clustered_bootstrap
from src.onset.run_candle_lgbm import _rank_ic, long_short, RECENT


def summarize_ablation(by_set: dict) -> dict:
    """Given {set_name: pooled long_short dict}, compute incremental net of
    adding candle geometry over the factor baseline."""
    f = by_set.get("factors", {}).get("net_sharpe")
    fc = by_set.get("factors_plus_candle", {}).get("net_sharpe")
    out = {"net_sharpe": {k: v.get("net_sharpe") for k, v in by_set.items()}}
    if f is not None and fc is not None:
        out["incremental_net_sharpe_from_candle"] = fc - f
    return out


def _lgbm(Xtr, ytr, Xte):
    d = lgb.Dataset(Xtr, label=ytr)
    b = lgb.train(dict(objective="regression", learning_rate=0.05, num_leaves=15,
                       min_data_in_leaf=50, verbose=-1, seed=42), d, num_boost_round=120)
    return b.predict(Xte)


def run_real() -> dict:
    df = pd.read_parquet(D1); df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["_fwd_r5"] = df.groupby("ts_code")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
    # EXCLUDE the forward target (and any _fwd* column) from the factor set --
    # select_feature_columns does not exclude it, which would leak the label.
    factor_cols = [c for c in select_feature_columns(df) if not c.startswith("_fwd")]
    feats = panel_candle_features(df, prior=9)
    pf = df.copy()
    for c in FEATURE_COLS:
        pf[c] = feats[c].values

    def candle_flat(anchors):
        X, m = build_anchor_sequences(pf, anchors[["ts_code", "trade_date"]], FEATURE_COLS, RECENT)
        return X.reshape(len(X), RECENT * len(FEATURE_COLS)), m

    sets = {"factors": [], "factors_plus_candle": [], "candle": []}
    per_split_ls = {name: {} for name in sets}
    rng = np.random.default_rng(42)
    d1_idx = df.set_index(["ts_code", "trade_date"])
    for sid in (1, 2, 3):
        ts, te_, ve, tee = SPLITS[sid]
        trp = df[(df["trade_date"] >= ts) & (df["trade_date"] < te_) & df["_fwd_r5"].notna()]
        tr = trp.sample(min(15000, len(trp)), random_state=int(rng.integers(0, 1e6))).reset_index(drop=True)
        wf = pd.read_parquet(ROOT / f"data/processed/wf_samples_split{sid}.parquet")
        wf["trade_date"] = wf["trade_date"].astype(str); wf = wf[wf["_fwd_r5"].notna()].copy()

        Ctr, mtr = candle_flat(tr); Cte, mte = candle_flat(wf)
        tr = tr[mtr].reset_index(drop=True); Ctr = Ctr[mtr]
        wf = wf[mte].reset_index(drop=True); Cte = Cte[mte]
        tr["mkt_neutral"], _ = market_neutral(tr); wf["mkt_neutral"], _ = market_neutral(wf)
        Ftr = tr[factor_cols].fillna(0.0).values
        Fte = d1_idx.reindex(list(zip(wf["ts_code"], wf["trade_date"])))[factor_cols].fillna(0.0).values

        feat_arrays = {"factors": (Ftr, Fte),
                       "factors_plus_candle": (np.hstack([Ftr, Ctr]), np.hstack([Fte, Cte])),
                       "candle": (Ctr, Cte)}
        for name, (Xtr, Xte) in feat_arrays.items():
            pred = _lgbm(Xtr, tr["mkt_neutral"].values, Xte)
            w = wf[["trade_date", "_fwd_r5", "mkt_neutral"]].copy(); w["sig"] = pred
            sets[name].append(w)
            per_split_ls[name][f"split{sid}"] = long_short(w)

    by_set = {}
    for name, parts in sets.items():
        allp = pd.concat(parts, ignore_index=True)
        by_set[name] = {"rank_ic_market_neutral": clustered_bootstrap(
            _rank_ic, allp["sig"].values, allp["mkt_neutral"].values,
            allp["trade_date"].astype(str).values, n_boot=300), **long_short(allp)}
    for name in by_set:
        by_set[name]["per_split_net_sharpe"] = {k: v.get("net_sharpe") for k, v in per_split_ls[name].items()}
    out = {"by_set": by_set, "summary": summarize_ablation(by_set)}
    (ROOT / "results/candle").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/candle/ablation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
