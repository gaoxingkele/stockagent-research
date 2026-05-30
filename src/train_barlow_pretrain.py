"""Pretrain TCN+CrossAttn encoder with Barlow Twins SSL on unlabeled D1.

After pretraining, the encoder weights are saved and can be loaded by
src/train_tcn_finetune.py for downstream supervised classification.

Usage:
    python -m src.train_barlow_pretrain --n-unlabeled 100000 --max-epochs 10
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.tcn_cross_attn import (
    TCNCrossAttnConfig, build_anchor_sequences,
)
from src.models.barlow_twins import (
    BarlowTwinsConfig, pretrain,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
D1 = ROOT / "data/processed/ashares_tushare_v1.parquet"
OUT = ROOT / "results/barlow_pretrain"
OUT.mkdir(parents=True, exist_ok=True)

NON_FEAT = {
    "ts_code", "trade_date", "industry",
    "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount",
    "r5", "r10", "r20", "r30", "r40",
    "dd5", "dd10", "dd20", "dd30", "dd40",
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
    ap.add_argument("--train-end", default="20250101",
                    help="Only pretrain on anchors before this date (avoid lookahead).")
    ap.add_argument("--n-unlabeled", type=int, default=100_000)
    ap.add_argument("--time-steps", type=int, default=30)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--projector-dim", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-off", type=float, default=5e-3)
    ap.add_argument("--noise-std", type=float, default=0.10)
    ap.add_argument("--dropout-p", type=float, default=0.15)
    ap.add_argument("--time-warp", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rng = np.random.default_rng(args.seed)

    logger.info("Loading D1...")
    df = pd.read_parquet(D1)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    feature_cols = select_feature_columns(df)
    logger.info(f"Features: {len(feature_cols)}")

    # Sample unlabeled anchors from pre-training window (before train_end)
    pool = df[df["trade_date"] < args.train_end]
    n = min(args.n_unlabeled, len(pool))
    sampled = pool.sample(n, random_state=int(rng.integers(0, 10**6)))
    logger.info(f"Sampled {n:,} unlabeled anchors from {len(pool):,} pool")

    logger.info("Building anchor sequences...")
    X, mask = build_anchor_sequences(df, sampled[["ts_code", "trade_date"]], feature_cols, args.time_steps)
    X = X[mask]
    logger.info(f"Pretrain set: {len(X):,} sequences × {args.time_steps} × {len(feature_cols)}")

    enc_cfg = TCNCrossAttnConfig(
        num_features=len(feature_cols),
        time_steps=args.time_steps,
        d_model=args.d_model,
    )
    bt_cfg = BarlowTwinsConfig(
        projector_dim=args.projector_dim,
        lambda_off=args.lambda_off,
        aug_noise_std=args.noise_std,
        aug_dropout=args.dropout_p,
        aug_time_warp_max=args.time_warp,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )

    encoder, stats = pretrain(X, enc_cfg, bt_cfg)

    out_dir = OUT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), out_dir / "encoder.pt")
    np.savez(out_dir / "norm.npz",
              mean=np.array(stats["mean"], dtype=np.float32),
              std=np.array(stats["std"], dtype=np.float32))
    # Stats minus large arrays
    save_stats = {k: v for k, v in stats.items() if k not in ("mean", "std")}
    save_stats["n_unlabeled"] = int(len(X))
    save_stats["feature_cols"] = feature_cols
    with open(out_dir / "pretrain_stats.json", "w") as f:
        json.dump(save_stats, f, indent=2, default=str)
    logger.info(f"Saved pretrained encoder to {out_dir}")


if __name__ == "__main__":
    main()
