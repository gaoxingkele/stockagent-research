"""T-008 (real data) — regime-invariant neural onset model on the real A-share
walk-forward data, using the REAL TCN encoder (not the TinyEncoder stub).

Environments for IRM = walk-forward split-1 and split-2 training windows.
Held-out evaluation = split-3 walk-forward test anchors (wf_samples_split3),
scored against the real forward 5-day return (_fwd_r5) with a date-CLUSTERED
bootstrap RankIC (the leakage-safe, cluster-robust protocol of T-005).

This is the paper's reference method instantiated on real data. The signal is
known to be weak; the deliverable is an honest, leakage-safe, cluster-robust
number from a correctly-wired pipeline — not alpha.

Run: .venv-xpu\\Scripts\\python.exe -m src.train_onset_real
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.tcn_cross_attn import (
    TCNCrossAttnConfig, TCNCrossAttnPatternCore, _auto_device,
    build_anchor_sequences, label_to_class,
)
from src.train_tcn_wf import D1, ROOT, SPLITS, select_feature_columns
from src.labels.fixed_horizon import fixed_horizon_label
from src.models.onset_intensity import OnsetIntensityHead, discrete_time_nll
from src.training.irm_onset import irm_penalty
from src.training.objectives import soft_rank_ic_loss
from src.evaluation.onset_eval import clustered_bootstrap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

T = 30
D_MODEL = 64
N_PER_ENV = 6000
N_STEPS = 200
BATCH = 256
SEED = 42


def _rank_ic(pred, target):
    pr = np.argsort(np.argsort(pred)); tr = np.argsort(np.argsort(target))
    return float(np.corrcoef(pr, tr)[0, 1])


def _build_env(df, feature_cols, start, end, n, rng):
    pool = df[(df["trade_date"] >= start) & (df["trade_date"] < end) & (df["_label"] != -127)]
    anchors = pool.sample(min(n, len(pool)), random_state=int(rng.integers(0, 10**6)))
    X, mask = build_anchor_sequences(df, anchors[["ts_code", "trade_date"]], feature_cols, T)
    cls = label_to_class(anchors["_label"])           # 0=down,1=neutral,2=up
    X, cls = X[mask], cls[mask]
    return X.astype("float32"), cls.astype("int64")


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    device = _auto_device()
    logger.info(f"device={device}")

    logger.info("Loading D1...")
    df = pd.read_parquet(D1)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    feature_cols = select_feature_columns(df)
    df["_label"] = fixed_horizon_label(df, horizon=5, threshold_up=0.03)
    logger.info(f"features={len(feature_cols)}")

    # IRM environments = split-1 and split-2 train windows
    envs_raw = []
    for sid in (1, 2):
        start, end = SPLITS[sid][0], SPLITS[sid][1]
        X, cls = _build_env(df, feature_cols, start, end, N_PER_ENV, rng)
        envs_raw.append((X, cls))
        logger.info(f"  env split{sid}: {len(X)} sequences")

    # standardize on pooled train stats
    allX = np.concatenate([e[0] for e in envs_raw], axis=0)
    mean = allX.mean(axis=(0, 1), keepdims=True)
    std = allX.std(axis=(0, 1), keepdims=True) + 1e-6

    def to_env(X, cls):
        Xn = (X - mean) / std
        events = np.zeros((len(X), T), dtype="float32")
        events[:, -1] = (cls == 2).astype("float32")          # onset = up move at last step
        up_bin = (cls == 2).astype("float32")
        ordinal = cls.astype("float32")                        # down<neutral<up ranking target
        return (torch.from_numpy(Xn), torch.from_numpy(events),
                torch.from_numpy(up_bin), torch.from_numpy(ordinal))

    envs = [to_env(X, cls) for X, cls in envs_raw]

    cfg = TCNCrossAttnConfig(num_features=len(feature_cols), time_steps=T, d_model=D_MODEL)
    enc = TCNCrossAttnPatternCore(cfg).to(device)
    head = OnsetIntensityHead(d_model=D_MODEL).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)

    logger.info("Training...")
    last = {}
    for step in range(N_STEPS):
        opt.zero_grad()
        nll_s = pen_s = rank_s = torch.zeros((), device=device)
        for X, events, up_bin, ordinal in envs:
            idx = torch.randint(0, len(X), (BATCH,))
            xb = X[idx].to(device); ev = events[idx].to(device)
            ub = up_bin[idx].to(device); od = ordinal[idx].to(device)
            lam = head(enc.forward_features(xb))           # [B,T] intensity
            score = lam.mean(dim=1)
            nll_s = nll_s + discrete_time_nll(lam, ev)
            pen_s = pen_s + irm_penalty(score, ub)
            rank_s = rank_s + soft_rank_ic_loss(score, od)
        n = len(envs)
        total = nll_s / n + pen_s / n + rank_s / n
        total.backward(); opt.step()
        if step % 50 == 0 or step == N_STEPS - 1:
            last = {"step": step, "total": float(total), "nll": float(nll_s / n),
                    "irm": float(pen_s / n), "rank": float(rank_s / n)}
            logger.info(f"  step {step}: {last}")

    # held-out eval on split-3 walk-forward test anchors vs real forward return
    logger.info("Evaluating on split-3 held-out test...")
    wf = pd.read_parquet(ROOT / "data/processed/wf_samples_split3.parquet")
    wf["trade_date"] = wf["trade_date"].astype(str)
    Xte, mask = build_anchor_sequences(df, wf[["ts_code", "trade_date"]], feature_cols, T)
    Xte = ((Xte - mean) / std).astype("float32")[mask]
    wf_v = wf[mask].reset_index(drop=True)
    enc.eval(); head.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(Xte), 1024):
            xb = torch.from_numpy(Xte[i:i + 1024]).to(device)
            scores.append(head(enc.forward_features(xb)).mean(dim=1).cpu().numpy())
    score = np.concatenate(scores)
    valid = wf_v["_fwd_r5"].notna().values
    ic = clustered_bootstrap(_rank_ic, score[valid], wf_v["_fwd_r5"].values[valid],
                             wf_v["trade_date"].values[valid], n_boot=500)

    stats = {
        "device": str(device), "encoder": "TCNCrossAttnPatternCore (real)",
        "features": len(feature_cols), "T": T, "d_model": D_MODEL,
        "n_per_env": N_PER_ENV, "n_steps": N_STEPS, "envs": ["split1_train", "split2_train"],
        "final_losses": last,
        "heldout": "split3_wf_test", "n_heldout": int(valid.sum()),
        "heldout_rank_ic_vs_fwd_r5": ic,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = ROOT / "results/onset/real"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"RankIC (clustered) = {ic['mean']:+.4f} [{ic['lo']:+.4f}, {ic['hi']:+.4f}]")
    logger.info(f"Saved -> {out/'stats.json'} ({stats['elapsed_sec']}s)")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
