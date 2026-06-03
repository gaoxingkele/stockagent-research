"""K3 -- REAL: hand-engineered candle features -> LightGBM, honest all-split
market-neutral cost-aware alpha test (path A).

For each walk-forward split: train LGBM on the split's train-window candle
features to predict the market-neutral forward return; predict that split's
wf_samples test anchors; report per-split AND pooled market-neutral RankIC +
long-short Sharpe GROSS and NET of 0.2%/leg cost, with date-clustered CIs.
The SIGN-K1 verdict is computed in K6.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_candle_lgbm
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.train_tcn_wf import D1, ROOT, SPLITS
from src.onset.candle_geometry import panel_candle_features, FEATURE_COLS
from src.models.tcn_cross_attn import build_anchor_sequences
from src.identify.neutral_targets import market_neutral
from src.evaluation.onset_eval import clustered_bootstrap

OUT = ROOT / "results/candle"
RECENT = 3
PRIOR = 9
COST_PER_LEG = 0.002          # 0.2% round-trip per leg; long-short = 2 legs
PERIODS_PER_YEAR = 252 / 5.0


def _rank_ic(pred, target):
    pr = np.argsort(np.argsort(pred)); tr = np.argsort(np.argsort(target))
    return float(np.corrcoef(pr, tr)[0, 1])


def long_short(df, sig="sig", ret="_fwd_r5", date="trade_date", frac=0.2,
               cost=2 * COST_PER_LEG, n_boot=500, seed=42):
    def ls_period(s):
        s = s.dropna(subset=[sig, ret])
        if len(s) < 5:
            return np.nan
        k = max(1, int(len(s) * frac)); o = s.sort_values(sig)
        return o[ret].iloc[-k:].mean() - o[ret].iloc[:k].mean()
    g = df.groupby(date).apply(ls_period).dropna()
    if len(g) < 2:
        return {"n_dates": int(len(g))}
    def sharpe(x):
        sd = x.std(); return float(np.sqrt(PERIODS_PER_YEAR) * x.mean() / sd) if sd > 0 else float("nan")
    net = g - cost
    rng = np.random.default_rng(seed); v = net.values; m = len(v)
    boot = [v[rng.integers(0, m, m)].mean() for _ in range(n_boot)]
    return {"n_dates": int(len(g)),
            "gross_mean": float(g.mean()), "gross_sharpe": sharpe(g),
            "net_mean": float(net.mean()), "net_sharpe": sharpe(net),
            "net_mean_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def fit_eval_candle(Xtr, ytr, Xte, te_df) -> dict:
    """te_df must have columns: sig (filled here), _fwd_r5, mkt_neutral, trade_date."""
    d = lgb.Dataset(Xtr, label=ytr)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=15,
                  min_data_in_leaf=50, verbose=-1, seed=42)
    booster = lgb.train(params, d, num_boost_round=120)
    te_df = te_df.copy(); te_df["sig"] = booster.predict(Xte)
    ic = clustered_bootstrap(_rank_ic, te_df["sig"].values, te_df["mkt_neutral"].values,
                             te_df["trade_date"].astype(str).values, n_boot=400)
    return {"rank_ic_market_neutral": ic, "long_short": long_short(te_df), "booster": booster}


def _flat(pf, anchors):
    X, mask = build_anchor_sequences(pf, anchors[["ts_code", "trade_date"]], FEATURE_COLS, RECENT)
    return X.reshape(len(X), RECENT * len(FEATURE_COLS)), mask


def run_real() -> dict:
    df = pd.read_parquet(D1); df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["_fwd_r5"] = df.groupby("ts_code")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
    feats = panel_candle_features(df, prior=PRIOR)
    pf = df.copy()
    for c in FEATURE_COLS:
        pf[c] = feats[c].values

    per_split = {}; pooled = []
    rng = np.random.default_rng(42)
    for sid in (1, 2, 3):
        ts, te_, ve, tee = SPLITS[sid]
        tr_pool = df[(df["trade_date"] >= ts) & (df["trade_date"] < te_) & df["_fwd_r5"].notna()]
        tr = tr_pool.sample(min(20000, len(tr_pool)), random_state=int(rng.integers(0, 1e6)))
        tr["mkt_neutral"], _ = market_neutral(tr)
        Xtr, mtr = _flat(pf, tr); tr = tr[mtr]
        Xtr = Xtr[mtr]

        wf = pd.read_parquet(ROOT / f"data/processed/wf_samples_split{sid}.parquet")
        wf["trade_date"] = wf["trade_date"].astype(str)
        wf = wf[wf["_fwd_r5"].notna()].copy()
        wf["mkt_neutral"], _ = market_neutral(wf)
        Xte, mte = _flat(pf, wf); wf = wf[mte]; Xte = Xte[mte]

        r = fit_eval_candle(Xtr, tr["mkt_neutral"].values, Xte, wf)
        booster = r.pop("booster")
        per_split[f"split{sid}"] = r
        w = wf[["trade_date", "_fwd_r5", "mkt_neutral"]].copy(); w["sig"] = booster.predict(Xte)
        pooled.append(w)

    allp = pd.concat(pooled, ignore_index=True)
    pooled_res = {"rank_ic_market_neutral": clustered_bootstrap(
        _rank_ic, allp["sig"].values, allp["mkt_neutral"].values,
        allp["trade_date"].astype(str).values, n_boot=400),
        "long_short": long_short(allp)}
    out = {"recent": RECENT, "prior": PRIOR, "cost_per_leg": COST_PER_LEG,
           "per_split": per_split, "pooled": pooled_res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lgbm.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
