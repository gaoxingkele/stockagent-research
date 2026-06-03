"""K4 -- REAL (XPU): learned candle-geometry SEQUENCE encoder, same honest
all-split market-neutral cost-aware eval, multi-seed (path B).

Feeds the last ~12 bars of candle geometry (K2 anchor_sequences) into a small
GRU regressor predicting the market-neutral forward return; per split trains
several seeds and ensembles; evaluates per-split + pooled (market-neutral
RankIC + GROSS/NET long-short, clustered). Honest expectation: at best a small
signal that does not clear the SIGN-K1 cost bar.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_candle_seq
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.train_tcn_wf import D1, ROOT, SPLITS
from src.models.tcn_cross_attn import _auto_device, build_anchor_sequences
from src.onset.candle_geometry import panel_candle_features, FEATURE_COLS
from src.identify.neutral_targets import market_neutral
from src.evaluation.onset_eval import clustered_bootstrap
from src.onset.run_candle_lgbm import _rank_ic, long_short

OUT = ROOT / "results/candle"
WINDOW = 12
PRIOR = 9
SEEDS = (0, 1, 2)


class SeqReg(nn.Module):
    def __init__(self, num_features, d_model=24):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x):
        h, _ = self.gru(torch.relu(self.proj(x)))
        return self.head(h[:, -1, :]).squeeze(-1)


def fit_eval_seq(Xtr, ytr, Xte, te_df, device="cpu", steps=150, batch=256, seeds=SEEDS) -> dict:
    """Train `len(seeds)` GRU regressors, ensemble predictions, evaluate."""
    F = Xtr.shape[-1]
    st = torch.as_tensor(Xtr, dtype=torch.float32); yy = torch.as_tensor(ytr, dtype=torch.float32)
    xte = torch.as_tensor(Xte, dtype=torch.float32).to(device)
    preds = []
    for s in seeds:
        torch.manual_seed(s)
        m = SeqReg(F).to(device); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        n = len(st)
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch, n),))
            opt.zero_grad()
            loss = ((m(st[idx].to(device)) - yy[idx].to(device)) ** 2).mean()
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            preds.append(m(xte).cpu().numpy())
    pred = np.mean(preds, axis=0)
    te_df = te_df.copy(); te_df["sig"] = pred
    ic = clustered_bootstrap(_rank_ic, pred, te_df["mkt_neutral"].values,
                             te_df["trade_date"].astype(str).values, n_boot=300)
    return {"rank_ic_market_neutral": ic, "long_short": long_short(te_df), "pred": pred}


def _seq(pf, anchors, mean, std):
    X, mask = build_anchor_sequences(pf, anchors[["ts_code", "trade_date"]], FEATURE_COLS, WINDOW)
    return ((X - mean) / std).astype("float32"), mask


def run_real() -> dict:
    t0 = time.time(); device = _auto_device()
    df = pd.read_parquet(D1); df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["_fwd_r5"] = df.groupby("ts_code")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
    feats = panel_candle_features(df, prior=PRIOR)
    pf = df.copy()
    for c in FEATURE_COLS:
        pf[c] = feats[c].values

    rng = np.random.default_rng(42); per_split = {}; pooled = []
    for sid in (1, 2, 3):
        ts, te_, ve, tee = SPLITS[sid]
        trp = df[(df["trade_date"] >= ts) & (df["trade_date"] < te_) & df["_fwd_r5"].notna()]
        tr = trp.sample(min(6000, len(trp)), random_state=int(rng.integers(0, 1e6)))
        Xtr0, m0 = build_anchor_sequences(pf, tr[["ts_code", "trade_date"]], FEATURE_COLS, WINDOW)
        mean = Xtr0[m0].mean(axis=(0, 1), keepdims=True); std = Xtr0[m0].std(axis=(0, 1), keepdims=True) + 1e-6
        tr = tr[m0]; tr["mkt_neutral"], _ = market_neutral(tr)
        Xtr = ((Xtr0[m0] - mean) / std).astype("float32")

        wf = pd.read_parquet(ROOT / f"data/processed/wf_samples_split{sid}.parquet")
        wf["trade_date"] = wf["trade_date"].astype(str); wf = wf[wf["_fwd_r5"].notna()].copy()
        Xte, mte = _seq(pf, wf, mean, std); wf = wf[mte]; Xte = Xte[mte]
        wf["mkt_neutral"], _ = market_neutral(wf)

        r = fit_eval_seq(Xtr, tr["mkt_neutral"].values, Xte, wf, device=device)
        pred = r.pop("pred"); per_split[f"split{sid}"] = r
        w = wf[["trade_date", "_fwd_r5", "mkt_neutral"]].copy(); w["sig"] = pred
        pooled.append(w)

    allp = pd.concat(pooled, ignore_index=True)
    pooled_res = {"rank_ic_market_neutral": clustered_bootstrap(
        _rank_ic, allp["sig"].values, allp["mkt_neutral"].values,
        allp["trade_date"].astype(str).values, n_boot=300), "long_short": long_short(allp)}
    out = {"device": str(device), "window": WINDOW, "seeds": list(SEEDS),
           "per_split": per_split, "pooled": pooled_res, "elapsed_sec": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "seq.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
