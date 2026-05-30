"""Walk-forward TCN+Cross-Attention training (Pathway 2 Pattern Core).

Trains one model per walk-forward split, then scores all 2000 anchors per split.
Outputs predictions compatible with the existing hybrid_router and eval scripts.

Usage:
    python -m src.train_tcn_wf --split 1 --max-epochs 5
    python -m src.train_tcn_wf --split all --max-epochs 10
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

from src.models.tcn_cross_attn import (
    TCNCrossAttnConfig, TCNCrossAttnPatternCore,
    build_anchor_sequences, train_model, predict,
    label_to_class, class_to_signal,
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


def train_one_split(split_id: int, args):
    train_start, train_end, val_end, test_end = SPLITS[split_id]
    logger.info(f"=== Split {split_id}: train {train_start}-{train_end}, val {train_end}-{val_end}, test {val_end}-{test_end} ===")

    logger.info("Loading D1 (may take ~10s)...")
    df = pd.read_parquet(D1)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    feature_cols = select_feature_columns(df)
    logger.info(f"  features: {len(feature_cols)}")

    # Compute labels
    y = fixed_horizon_label(df, horizon=5, threshold_up=0.03)
    df["_label"] = y

    # Time splits (anchors)
    tr_mask = (df["trade_date"] >= train_start) & (df["trade_date"] < train_end)
    va_mask = (df["trade_date"] >= train_end) & (df["trade_date"] < val_end)

    # For training/val, sample down to keep TCN tractable on CPU
    rng = np.random.default_rng(args.seed)
    tr_pool = df[tr_mask & (df["_label"] != -127)]
    va_pool = df[va_mask & (df["_label"] != -127)]

    n_train = min(args.n_train, len(tr_pool))
    n_val = min(args.n_val, len(va_pool))
    tr_anchors = tr_pool.sample(n_train, random_state=int(rng.integers(0, 10**6)))
    va_anchors = va_pool.sample(n_val, random_state=int(rng.integers(0, 10**6)))
    logger.info(f"  train anchors: {n_train} (from {len(tr_pool):,} pool)")
    logger.info(f"  val anchors:   {n_val} (from {len(va_pool):,} pool)")

    # Test anchors from prebuilt walk-forward sample (matches LGBM evaluation)
    wf_path = ROOT / f"data/processed/wf_samples_split{split_id}.parquet"
    te_anchors = pd.read_parquet(wf_path)
    logger.info(f"  test anchors:  {len(te_anchors)} (from {wf_path.name})")

    # Build sequences (T=30 days history per anchor)
    T = args.time_steps
    logger.info("Building train sequences...")
    X_tr, mask_tr = build_anchor_sequences(df, tr_anchors[["ts_code", "trade_date"]], feature_cols, T)
    y_tr = label_to_class(tr_anchors["_label"])
    X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]

    logger.info("Building val sequences...")
    X_va, mask_va = build_anchor_sequences(df, va_anchors[["ts_code", "trade_date"]], feature_cols, T)
    y_va = label_to_class(va_anchors["_label"])
    X_va, y_va = X_va[mask_va], y_va[mask_va]

    logger.info("Building test sequences...")
    te_keys = te_anchors[["ts_code", "trade_date"]].copy()
    te_keys["trade_date"] = te_keys["trade_date"].astype(str)
    X_te, mask_te = build_anchor_sequences(df, te_keys, feature_cols, T)
    logger.info(f"  test sequences valid: {mask_te.sum()}/{len(te_keys)}")

    # Train
    cfg = TCNCrossAttnConfig(
        num_features=len(feature_cols),
        time_steps=T,
        d_model=args.d_model,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        seed=args.seed,
    )
    logger.info(f"TCNCrossAttn config: d_model={cfg.d_model}, T={cfg.time_steps}, F={cfg.num_features}")
    model, stats = train_model(X_tr, y_tr, X_va, y_va, cfg=cfg)

    # Predict on test
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    probs = predict(model, X_te[mask_te], mean, std, batch_size=cfg.batch_size)
    signal = class_to_signal(probs)

    # Pack: align to original test anchor order, nan-fill where mask was False
    n_te = len(te_keys)
    pred_p_down = np.full(n_te, np.nan)
    pred_p_neutral = np.full(n_te, np.nan)
    pred_p_up = np.full(n_te, np.nan)
    pred_signal = np.full(n_te, np.nan)
    valid_pos = np.where(mask_te)[0]
    pred_p_down[valid_pos] = probs[:, 0]
    pred_p_neutral[valid_pos] = probs[:, 1]
    pred_p_up[valid_pos] = probs[:, 2]
    pred_signal[valid_pos] = signal

    out = te_anchors.copy()
    out["tcn_p_down"] = pred_p_down
    out["tcn_p_neutral"] = pred_p_neutral
    out["tcn_p_up"] = pred_p_up
    out["tcn_pump_ratio"] = pred_signal

    out_dir = RESULTS / f"tcn_wf_split{split_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "predictions.parquet", index=False)
    with open(out_dir / "stats.json", "w") as f:
        json.dump({
            "split_id": split_id, "config": asdict(cfg),
            "stats": {k: v for k, v in stats.items() if k != "mean" and k != "std"},
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test_total": int(n_te),
            "n_test_valid": int(mask_te.sum()),
        }, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model.pt")
    np.savez(out_dir / "norm.npz", mean=mean, std=std)
    logger.info(f"Saved -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="1", help="Split ID (1/2/3) or 'all'")
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=20_000)
    ap.add_argument("--time-steps", type=int, default=30)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    splits = [1, 2, 3] if args.split == "all" else [int(args.split)]
    for s in splits:
        train_one_split(s, args)


if __name__ == "__main__":
    main()
