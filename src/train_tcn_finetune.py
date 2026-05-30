"""Finetune a Barlow-Twins-pretrained encoder on the supervised onset task.

Compares Pathway 3 (SSL pretrain + finetune) to Pathway 2 (TCN from scratch).

Usage:
    python -m src.train_tcn_finetune --pretrain-tag v1 --split 1 \\
        --n-train 100000 --max-epochs 5 --tag pwy3
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tcn_cross_attn import (
    TCNCrossAttnPatternCore, TCNCrossAttnConfig,
    build_anchor_sequences, label_to_class, class_to_signal,
    predict,
)
from src.models.barlow_twins import (
    BarlowTwinsEncoder, init_pattern_core_from_pretrained,
)
from src.labels.fixed_horizon import fixed_horizon_label

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
D1 = ROOT / "data/processed/ashares_tushare_v1.parquet"
RESULTS = ROOT / "results"

NON_FEAT = {
    "ts_code", "trade_date", "industry",
    "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount",
    "r5", "r10", "r20", "r30", "r40",
    "dd5", "dd10", "dd20", "dd30", "dd40",
}

SPLITS = {
    1: ("20220104", "20250101", "20250401", "20250701"),
    2: ("20220104", "20250401", "20250701", "20251001"),
    3: ("20220104", "20250701", "20251001", "20260101"),
}


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in NON_FEAT:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].notna().mean() < 0.05:
            continue
        cols.append(c)
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-tag", default="v1", help="Subdir under results/barlow_pretrain/")
    ap.add_argument("--split", type=str, default="1")
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=20_000)
    ap.add_argument("--time-steps", type=int, default=30)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr-factor", type=float, default=0.1,
                    help="Multiply LR for pretrained encoder weights (lower → less drift)")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="pwy3")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Freeze pretrained encoder; only train classifier")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pretrain_dir = RESULTS / "barlow_pretrain" / args.pretrain_tag
    if not (pretrain_dir / "encoder.pt").exists():
        raise FileNotFoundError(f"Pretrained encoder not found at {pretrain_dir}")
    with open(pretrain_dir / "pretrain_stats.json") as f:
        pretrain_stats = json.load(f)
    feature_cols_pretrain = pretrain_stats["feature_cols"]
    pretrain_norm = np.load(pretrain_dir / "norm.npz")
    mean_pre, std_pre = pretrain_norm["mean"], pretrain_norm["std"]
    logger.info(f"Loaded pretrained encoder from {pretrain_dir}")

    split_id = int(args.split)
    train_start, train_end, val_end, test_end = SPLITS[split_id]
    logger.info(f"=== Split {split_id} ===")

    logger.info("Loading D1...")
    df = pd.read_parquet(D1)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    feature_cols = select_feature_columns(df)
    if feature_cols != feature_cols_pretrain:
        logger.warning("Pretrain feature_cols differ from current — re-using current set")

    y = fixed_horizon_label(df, horizon=5, threshold_up=0.03)
    df["_label"] = y

    rng = np.random.default_rng(args.seed)
    tr_pool = df[(df["trade_date"] >= train_start) & (df["trade_date"] < train_end) & (df["_label"] != -127)]
    va_pool = df[(df["trade_date"] >= train_end) & (df["trade_date"] < val_end) & (df["_label"] != -127)]
    tr_anchors = tr_pool.sample(min(args.n_train, len(tr_pool)), random_state=int(rng.integers(0, 10**6)))
    va_anchors = va_pool.sample(min(args.n_val, len(va_pool)), random_state=int(rng.integers(0, 10**6)))

    wf_path = ROOT / f"data/processed/wf_samples_split{split_id}.parquet"
    te_anchors = pd.read_parquet(wf_path)
    te_anchors["trade_date"] = te_anchors["trade_date"].astype(str)

    T = args.time_steps
    logger.info("Building sequences...")
    X_tr, m_tr = build_anchor_sequences(df, tr_anchors[["ts_code", "trade_date"]], feature_cols, T)
    X_va, m_va = build_anchor_sequences(df, va_anchors[["ts_code", "trade_date"]], feature_cols, T)
    X_te, m_te = build_anchor_sequences(df, te_anchors[["ts_code", "trade_date"]], feature_cols, T)
    y_tr = label_to_class(tr_anchors["_label"])[m_tr]
    y_va = label_to_class(va_anchors["_label"])[m_va]
    X_tr, X_va = X_tr[m_tr], X_va[m_va]
    logger.info(f"train={len(X_tr):,}, val={len(X_va):,}, test_valid={m_te.sum()}/{len(te_anchors)}")

    # Normalize with pretrain stats (consistent with pretraining)
    X_tr = ((X_tr - mean_pre) / std_pre).clip(-5.0, 5.0).astype(np.float32)
    X_va = ((X_va - mean_pre) / std_pre).clip(-5.0, 5.0).astype(np.float32)
    X_te = ((X_te - mean_pre) / std_pre).clip(-5.0, 5.0).astype(np.float32)

    # Build Pattern Core and load pretrained encoder weights
    cfg = TCNCrossAttnConfig(num_features=len(feature_cols), time_steps=T, d_model=args.d_model)
    base = TCNCrossAttnPatternCore(cfg)
    encoder = BarlowTwinsEncoder(base)
    encoder.load_state_dict(torch.load(pretrain_dir / "encoder.pt", map_location="cpu"))
    model = init_pattern_core_from_pretrained(encoder, cfg)
    logger.info("Loaded pretrained encoder into Pattern Core")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer with differential LR for encoder vs classifier
    if args.freeze_encoder:
        for p in model.tcn.parameters():       p.requires_grad = False
        for p in model.space_embed.parameters(): p.requires_grad = False
        for p in model.st_cross_attn.parameters(): p.requires_grad = False
        params = list(model.classifier.parameters())
        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    else:
        enc_params = list(model.tcn.parameters()) + list(model.space_embed.parameters()) + list(model.st_cross_attn.parameters())
        cls_params = list(model.classifier.parameters())
        optim = torch.optim.AdamW([
            {"params": enc_params, "lr": args.lr * args.encoder_lr_factor},
            {"params": cls_params, "lr": args.lr},
        ], weight_decay=1e-5)

    from torch.utils.data import DataLoader, TensorDataset
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.int64))),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va.astype(np.int64))),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    best_val, best_state, patience_left = float("inf"), None, args.patience
    t0 = time.time()
    for epoch in range(args.max_epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        train_loss = tot / max(n, 1)

        model.eval()
        vt, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vt += F.cross_entropy(logits, yb).item() * len(xb)
                vn += len(xb)
        val_loss = vt / max(vn, 1)
        logger.info(f"  epoch {epoch+1:>2}  train_ce={train_loss:.4f}  val_ce={val_loss:.4f}")
        if val_loss < best_val - 1e-4:
            best_val, best_state, patience_left = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(f"  early stopping at epoch {epoch+1}, best_val={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"  done in {time.time()-t0:.1f}s")

    # Predict on test
    probs = predict(model, X_te[m_te], mean_pre, std_pre, batch_size=args.batch_size)
    signal = class_to_signal(probs)
    n_te = len(te_anchors)
    pred = {k: np.full(n_te, np.nan) for k in ["pwy3_p_down", "pwy3_p_neutral", "pwy3_p_up", "pwy3_pump_ratio"]}
    valid_pos = np.where(m_te)[0]
    pred["pwy3_p_down"][valid_pos] = probs[:, 0]
    pred["pwy3_p_neutral"][valid_pos] = probs[:, 1]
    pred["pwy3_p_up"][valid_pos] = probs[:, 2]
    pred["pwy3_pump_ratio"][valid_pos] = signal

    out = te_anchors.copy()
    for k, v in pred.items():
        out[k] = v

    out_dir = RESULTS / f"pwy3_finetune_split{split_id}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "predictions.parquet", index=False)
    with open(out_dir / "stats.json", "w") as f:
        json.dump({
            "split_id": split_id, "best_val": best_val,
            "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
            "n_test_valid": int(m_te.sum()),
            "pretrain_tag": args.pretrain_tag,
            "encoder_lr_factor": args.encoder_lr_factor,
            "freeze_encoder": args.freeze_encoder,
            "config": asdict(cfg),
        }, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model.pt")
    logger.info(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
