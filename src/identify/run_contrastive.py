"""NB5 — REAL (XPU, $0): contrastive (idiosyncratic) vs raw target, evaluated
market-neutral, on real A-share walk-forward data.

Two arms, same ContrastiveEncoder (stock seq + per-date market-mean reference seq):
  RAW     : predict _fwd_r5
  NEUTRAL : predict the market-neutral residual (per-date demean of _fwd_r5)
Held-out (split3): market-neutral RankIC (pred vs neutral target) + tradable
long-short Sharpe (rank by pred, top-minus-bottom of raw _fwd_r5), clustered.

Honest expectation (SIGN-A1/R1): the idiosyncratic signal is SMALLER but more
regime-stable; shrinkage vs raw is not failure. Gate = machinery runs + finding.

Run: .venv-xpu\\Scripts\\python.exe -m src.identify.run_contrastive
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.tcn_cross_attn import _auto_device, build_anchor_sequences
from src.train_tcn_wf import D1, ROOT, SPLITS, select_feature_columns
from src.identify.contrastive_encoder import ContrastiveEncoder, neutral_mse
from src.identify.neutral_targets import market_neutral
from src.identify.run_neutral_identify import long_short
from src.evaluation.onset_eval import clustered_bootstrap

T = 30
OUT = ROOT / "results/identify/contrastive"


def _rank_ic(pred, target):
    pr = np.argsort(np.argsort(pred)); tr = np.argsort(np.argsort(target))
    return float(np.corrcoef(pr, tr)[0, 1])


def fit_eval(stock_tr, ref_tr, y_tr, stock_te, ref_te, neutral_te, raw_te, dates_te,
             device="cpu", steps=200, batch=256, seed=42) -> dict:
    """Train ContrastiveEncoder to predict y_tr; eval market-neutral on test."""
    torch.manual_seed(seed)
    F = stock_tr.shape[-1]
    m = ContrastiveEncoder(F).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    st = torch.as_tensor(stock_tr, dtype=torch.float32)
    rf = torch.as_tensor(ref_tr, dtype=torch.float32)
    yy = torch.as_tensor(y_tr, dtype=torch.float32)
    n = len(st)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        opt.zero_grad()
        pred = m(st[idx].to(device), rf[idx].to(device))
        loss = neutral_mse(pred, yy[idx].to(device))
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pr = m(torch.as_tensor(stock_te, dtype=torch.float32).to(device),
               torch.as_tensor(ref_te, dtype=torch.float32).to(device)).cpu().numpy()
    ic = clustered_bootstrap(_rank_ic, pr, neutral_te, dates_te, n_boot=300)
    df = pd.DataFrame({"sig": pr, "_fwd_r5": raw_te, "trade_date": dates_te})
    ls = long_short(df, "sig")
    return {"market_neutral_rank_ic": ic, "long_short": ls, "final_train_mse": float(loss.detach())}


def _build(df_panel, market_panel, anchors, feats, mean, std):
    Xs, ms = build_anchor_sequences(df_panel, anchors[["ts_code", "trade_date"]], feats, T)
    ref_keys = anchors[["trade_date"]].copy(); ref_keys["ts_code"] = "_MKT_"
    Xr, mr = build_anchor_sequences(market_panel, ref_keys[["ts_code", "trade_date"]], feats, T)
    mask = ms & mr
    return ((Xs[mask] - mean) / std).astype("float32"), ((Xr[mask] - mean) / std).astype("float32"), mask


def run_real() -> dict:
    t0 = time.time(); device = _auto_device()
    df = pd.read_parquet(D1); df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    feats = select_feature_columns(df)
    market_panel = df.groupby("trade_date")[feats].mean().reset_index()
    market_panel["ts_code"] = "_MKT_"

    def anchors_for(split, n):
        a = pd.read_parquet(ROOT / f"results/poc_wf_split{split}/predictions.parquet")
        a["trade_date"] = a["trade_date"].astype(str)
        return a[["ts_code", "trade_date", "_fwd_r5"]].dropna().head(n)

    tr = pd.concat([anchors_for(1, 4000), anchors_for(2, 4000)], ignore_index=True)
    te = anchors_for(3, 2000)
    # standardization stats from train stock sequences
    Xtr0, _ = build_anchor_sequences(df, tr[["ts_code", "trade_date"]], feats, T)
    mean = Xtr0.mean(axis=(0, 1), keepdims=True); std = Xtr0.std(axis=(0, 1), keepdims=True) + 1e-6

    Xs_tr, Xr_tr, mtr = _build(df, market_panel, tr, feats, mean, std)
    Xs_te, Xr_te, mte = _build(df, market_panel, te, feats, mean, std)
    tr_v, te_v = tr[mtr].reset_index(drop=True), te[mte].reset_index(drop=True)
    tr_v["mkt_neutral"], _ = market_neutral(tr_v); te_v["mkt_neutral"], _ = market_neutral(te_v)

    res = {"device": str(device), "n_train": int(len(tr_v)), "n_test": int(len(te_v)), "arms": {}}
    for arm, ycol in [("raw", "_fwd_r5"), ("neutral", "mkt_neutral")]:
        res["arms"][arm] = fit_eval(
            Xs_tr, Xr_tr, tr_v[ycol].values, Xs_te, Xr_te,
            te_v["mkt_neutral"].values, te_v["_fwd_r5"].values,
            te_v["trade_date"].astype(str).values, device=device)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
