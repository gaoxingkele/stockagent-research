"""MLP training entry — mirrors train.py for cross-architecture verification."""
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
from src.labels.triple_barrier import triple_barrier_label
from src.models.mlp_baseline import MLPConfig, class_to_signal, train
from src.evaluation.metrics import summary

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/ashares_tushare_v1.parquet"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLS = {
    "ts_code", "trade_date", "industry",
    "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount",
    "r5", "r10", "r20", "r30", "r40",
    "dd5", "dd10", "dd20", "dd30", "dd40",
    "_ret_1",
}


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].notna().mean() < 0.05:
            continue
        cols.append(c)
    return cols


def compute_forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    p = df["close"]
    g = df.groupby("ts_code")["close"]
    return g.shift(-horizon) / p - 1.0


def split_by_date(df, train_end, val_end, test_end, train_start=None):
    d = df["trade_date"].astype(str)
    train_mask = d < train_end
    if train_start is not None:
        train_mask = train_mask & (d >= train_start)
    val_mask = (d >= train_end) & (d < val_end)
    test_mask = (d >= val_end) & (d < test_end)
    return df.index[train_mask], df.index[val_mask], df.index[test_mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["fh", "pwc", "tb"], default="fh")
    ap.add_argument("--filter", choices=["none", "fixed_pwc", "adaptive_pwc"], default="none")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--pwc-past-window", type=int, default=5)
    ap.add_argument("--pwc-past-threshold", type=float, default=0.08)
    ap.add_argument("--tb-u", type=float, default=0.05)
    ap.add_argument("--tb-d", type=float, default=0.03)
    ap.add_argument("--tb-h", type=int, default=5)
    ap.add_argument("--adaptive-lookback", type=int, default=20)
    ap.add_argument("--adaptive-sigma", type=float, default=2.0)
    ap.add_argument("--adaptive-threshold", type=float, default=0.08)
    ap.add_argument("--train-start", default="20220104")
    ap.add_argument("--train-end", default="20250101")
    ap.add_argument("--val-end", default="20250701")
    ap.add_argument("--test-end", default="20260601")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info(f"Loading D1 from {DATA}")
    t0 = time.time()
    df = pd.read_parquet(DATA)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    logger.info(f"D1: {len(df):,} rows × {len(df.columns)} cols in {time.time()-t0:.1f}s")

    # Label
    if args.label == "fh":
        y = fixed_horizon_label(df, horizon=args.horizon, threshold_up=args.threshold)
        label_meta = {"type": "fh", "horizon": args.horizon, "threshold_up": args.threshold}
    elif args.label == "pwc":
        y = fixed_window_pwc_label(
            df, horizon=args.horizon, up_threshold=0.10, dd_threshold=0.05,
            past_window=args.pwc_past_window, past_threshold=args.pwc_past_threshold,
        )
        label_meta = {"type": "pwc", "horizon": args.horizon,
                      "past_threshold": args.pwc_past_threshold}
    else:
        y = triple_barrier_label(df, u=args.tb_u, d=args.tb_d, H=args.tb_h)
        label_meta = {"type": "tb", "u": args.tb_u, "d": args.tb_d, "H": args.tb_h}

    valid = (y != -127)
    logger.info(f"Label {args.label}: +1={(y==1).mean():.2%}, -1={(y==-1).mean():.2%}, valid={valid.mean():.2%}")

    # Sample filter
    sample_filter = None
    if args.filter == "fixed_pwc":
        g = df.groupby("ts_code")["close"]
        past_r = df["close"] / g.shift(args.pwc_past_window) - 1.0
        sample_filter = past_r <= args.pwc_past_threshold
        logger.info(f"fixed_pwc filter: kept {sample_filter.mean():.2%}")
    elif args.filter == "adaptive_pwc":
        from src.onset.simple_cpd import adaptive_cumret
        df["_ret_1"] = df.groupby("ts_code")["close"].pct_change(1)
        cum = adaptive_cumret(df, return_col="_ret_1",
                              lookback=args.adaptive_lookback, sigma_mult=args.adaptive_sigma)
        sample_filter = cum <= args.adaptive_threshold
        logger.info(f"adaptive_pwc filter: kept {sample_filter.mean():.2%}")

    features = select_feature_columns(df)
    logger.info(f"Using {len(features)} features")
    X = df[features]

    tr, va, te = split_by_date(df, args.train_end, args.val_end, args.test_end,
                                train_start=args.train_start)
    logger.info(f"Split before filter: train={len(tr):,} val={len(va):,} test={len(te):,}")
    tr = tr[y.loc[tr] != -127]
    va = va[y.loc[va] != -127]
    te = te[y.loc[te] != -127]
    if sample_filter is not None:
        before_tr, before_va = len(tr), len(va)
        tr = tr[sample_filter.loc[tr].fillna(False)]
        va = va[sample_filter.loc[va].fillna(False)]
        logger.info(f"After sample_filter: train {before_tr:,}->{len(tr):,} val {before_va:,}->{len(va):,}")

    # Train
    cfg = MLPConfig(
        hidden_dims=tuple(args.hidden), dropout=args.dropout,
        learning_rate=args.lr, batch_size=args.batch_size,
        max_epochs=args.max_epochs, early_stopping_patience=args.patience,
        seed=args.seed,
    )
    logger.info(f"MLPConfig: {asdict(cfg)}")
    _, test_probs = train(X.loc[tr], y.loc[tr], X.loc[va], y.loc[va], X.loc[te], cfg=cfg)
    signal = class_to_signal(test_probs)

    fwd = compute_forward_return(df, horizon=args.horizon)
    target = fwd.loc[te]
    valid_t = target.notna()
    signal_s = pd.Series(signal, index=te)[valid_t]
    target_s = target[valid_t]
    dates_s = df["trade_date"].loc[signal_s.index]

    metrics = summary(signal_s, target_s, dates_s, k=20)
    logger.info("=== Test metrics ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    suffix = "" if args.filter == "none" else f"_{args.filter}"
    name = args.name or f"mlp_{args.label}{suffix}_h{args.horizon}_seed{args.seed}_{int(time.time())}"
    run_dir = RESULTS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "mlp",
            "label": label_meta,
            "filter": args.filter,
            "n_features": len(features),
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "metrics": {k: float(v) if v is not None and not isinstance(v, str) else v
                        for k, v in metrics.items()},
            "config": asdict(cfg),
            "args": vars(args),
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
