"""Training entry point for stockagent-research experiments.

Minimal CLI (no Hydra yet — added later as the experiment matrix grows):

    python -m src.train --label fh --horizon 5 --threshold 0.03
    python -m src.train --label pwc

The script loads D1, generates the requested label, splits by date,
trains LightGBM, and reports RankIC / IR / TopK metrics on the test window.
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

from src.labels.fixed_horizon import fixed_horizon_label
from src.labels.fixed_window_pwc import fixed_window_pwc_label
from src.models.lgbm_baseline import LGBMConfig, predict_signal, train
from src.evaluation.metrics import summary

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/ashares_tushare_v1.parquet"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Columns that must NOT be used as features
NON_FEATURE_COLS = {
    "ts_code", "trade_date", "industry",
    "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount",
    # FORWARD-LOOKING in production factor_lab (leakage source, confirmed 2026-05-28):
    #   r5/r10/r20/r30/r40   = close.shift(-h)/close - 1  (NOT past return!)
    #   dd5/dd10/dd20/dd30/dd40 = forward drawdown
    # These are training-label assistants in production, never features.
    "r5", "r10", "r20", "r30", "r40",
    "dd5", "dd10", "dd20", "dd30", "dd40",
}


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Pick numeric factor columns, excluding raw OHLCV and identifiers."""
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        # Skip columns with > 95% missing in the available rows
        if df[c].notna().mean() < 0.05:
            continue
        cols.append(c)
    return cols


def compute_forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Forward simple return for evaluation."""
    p = df["close"]
    g = df.groupby("ts_code")["close"]
    return g.shift(-horizon) / p - 1.0


def split_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    test_end: str,
    train_start: str | None = None,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Return train/val/test index sets for a single time split."""
    d = df["trade_date"].astype(str)
    train_mask = d < train_end
    if train_start is not None:
        train_mask = train_mask & (d >= train_start)
    val_mask = (d >= train_end) & (d < val_end)
    test_mask = (d >= val_end) & (d < test_end)
    return df.index[train_mask], df.index[val_mask], df.index[test_mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["fh", "pwc"], default="fh")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.03,
                    help="FH up threshold (for fh) or up_threshold (for pwc)")
    ap.add_argument("--pwc-past-threshold", type=float, default=0.08)
    ap.add_argument("--train-start", default="20220104")
    ap.add_argument("--train-end", default="20250101")
    ap.add_argument("--val-end", default="20250701")
    ap.add_argument("--test-end", default="20260601")
    ap.add_argument("--num-boost-round", type=int, default=300)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="Run name (default: label+timestamp)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    t0 = time.time()
    logger.info(f"Loading D1 from {DATA}")
    df = pd.read_parquet(DATA)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    logger.info(f"D1 loaded: {len(df):,} rows × {len(df.columns)} cols in {time.time()-t0:.1f}s")

    # Label
    if args.label == "fh":
        y = fixed_horizon_label(df, horizon=args.horizon, threshold_up=args.threshold)
        label_meta = {"type": "fh", "horizon": args.horizon, "threshold_up": args.threshold}
    else:
        y = fixed_window_pwc_label(
            df, horizon=args.horizon,
            up_threshold=0.10, dd_threshold=0.05,
            past_window=5, past_threshold=args.pwc_past_threshold,
        )
        label_meta = {"type": "pwc", "horizon": args.horizon,
                      "up_threshold": 0.10, "dd_threshold": 0.05,
                      "past_window": 5, "past_threshold": args.pwc_past_threshold}

    valid_mask = (y != -127)
    pos_rate = (y == 1).mean()
    neg_rate = (y == -1).mean()
    logger.info(f"Label {args.label}: +1 = {pos_rate:.2%}, -1 = {neg_rate:.2%}, "
                f"valid = {valid_mask.mean():.2%}")

    # Features
    features = select_feature_columns(df)
    logger.info(f"Using {len(features)} features")

    X = df[features]

    # Split
    tr_idx, va_idx, te_idx = split_by_date(
        df, args.train_end, args.val_end, args.test_end, train_start=args.train_start
    )
    logger.info(f"Split: train={len(tr_idx):,} val={len(va_idx):,} test={len(te_idx):,}")

    # Restrict to rows with valid label
    tr_idx = tr_idx[y.loc[tr_idx] != -127]
    va_idx = va_idx[y.loc[va_idx] != -127]
    te_idx = te_idx[y.loc[te_idx] != -127]
    logger.info(f"After label filter: train={len(tr_idx):,} val={len(va_idx):,} test={len(te_idx):,}")

    # Train
    cfg = LGBMConfig(
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stop,
        seed=args.seed,
    )
    booster = train(
        X.loc[tr_idx], y.loc[tr_idx],
        X.loc[va_idx], y.loc[va_idx],
        cfg=cfg,
    )

    # Predict on test
    signal_test = predict_signal(booster, X.loc[te_idx])

    # Forward return for evaluation
    fwd = compute_forward_return(df, horizon=args.horizon)
    target_test = fwd.loc[te_idx]

    test_valid = target_test.notna()
    signal_test = pd.Series(signal_test, index=te_idx)[test_valid]
    target_test = target_test[test_valid]
    dates_test = df["trade_date"].loc[signal_test.index]

    # Metrics
    metrics = summary(signal_test, target_test, dates_test, k=20)
    logger.info("=== Test metrics ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    # Save
    name = args.name or f"{args.label}_h{args.horizon}_seed{args.seed}_{int(time.time())}"
    run_dir = RESULTS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "label": label_meta,
            "n_features": len(features),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "n_test": int(len(te_idx)),
            "metrics": {k: float(v) if v is not None and not isinstance(v, str) else v
                        for k, v in metrics.items()},
            "config": asdict(cfg),
            "args": vars(args),
        }, f, indent=2, ensure_ascii=False)
    booster.save_model(str(run_dir / "model.txt"))

    # B1.1 验收
    rank_ic_mean = metrics["rank_ic_mean"]
    rank_ic_ir = metrics["rank_ic_ir"]
    logger.info("=== B1.1 Acceptance ===")
    logger.info(f"  RankIC mean: {rank_ic_mean:.4f}  (require > 0.03)  -> {'PASS' if rank_ic_mean > 0.03 else 'FAIL'}")
    logger.info(f"  RankIC IR:   {rank_ic_ir:.4f}  (require > 0.5)   -> {'PASS' if rank_ic_ir > 0.5 else 'FAIL'}")

    logger.info(f"\nRun saved: {run_dir}")


if __name__ == "__main__":
    main()
