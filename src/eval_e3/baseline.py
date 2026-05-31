"""E3 — tabular price baseline on a FLARE stock-movement benchmark.

Trains a LightGBM classifier on the parsed most-recent-day price features
(the same momentum features the LLM sees in-context) and predicts the test
split. This is the "strong tabular baseline" the LLM must beat -- the exact
comparison LLM-finance papers make on these datasets.

Outputs results/e3_<dataset>/baseline_test.parquet with columns
[id, ticker, date, gold, baseline_p] (baseline_p = P(gold=1)).

Usage:
    .venv-xpu\\Scripts\\python.exe -m src.eval_e3.baseline --dataset acl
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/e3_flare"
RESULTS = ROOT / "results"

DROP = {"id", "ticker", "date", "gold", "answer", "query"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="acl")
    args = ap.parse_args()

    raw = BASE / args.dataset
    tr = pd.read_parquet(raw / "parsed_train.parquet")
    va = pd.read_parquet(raw / "parsed_valid.parquet")
    te = pd.read_parquet(raw / "parsed_test.parquet")
    feats = feature_cols(tr)
    print(f"{args.dataset}: train {len(tr)} / valid {len(va)} / test {len(te)} | {len(feats)} features")

    dtr = lgb.Dataset(tr[feats], label=tr["gold"])
    dva = lgb.Dataset(va[feats], label=va["gold"], reference=dtr)
    params = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                  num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, min_data_in_leaf=50, verbose=-1, seed=42)
    booster = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40, verbose=False)])

    p = booster.predict(te[feats], num_iteration=booster.best_iteration)
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(te["gold"], pred)
    mcc = matthews_corrcoef(te["gold"], pred)
    print(f"  TEST accuracy={acc:.4f}  MCC={mcc:+.4f}  "
          f"(majority-class baseline={max(te['gold'].mean(), 1-te['gold'].mean()):.4f})")

    out = te[["id", "ticker", "date", "gold"]].copy()
    out["baseline_p"] = p
    out_dir = RESULTS / f"e3_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "baseline_test.parquet", index=False)
    print(f"  wrote {out_dir / 'baseline_test.parquet'}")


if __name__ == "__main__":
    main()
