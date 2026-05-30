"""Evaluate TCN walk-forward predictions across the 3 splits.

Ranking score = tcn_pump_ratio (= p_up/(p_down+eps), same as LGBM Pattern Core).
Target        = _fwd_r5 (forward 5-day return).

Writes metrics.json into each results/tcn_wf_split{N}/ (so aggregate.py picks it
up) plus a pooled results/tcn_wf_eval/metrics.json, and prints a markdown table.

Usage:
    .venv-xpu\\Scripts\\python.exe -m src.evaluation.eval_tcn_wf
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from src.evaluation.metrics import summary

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SCORE = "tcn_pump_ratio"
TARGET = "_fwd_r5"
DATE = "trade_date"
K = 20


def eval_one(split_id: int) -> tuple[dict, pd.DataFrame]:
    run_dir = RESULTS / f"tcn_wf_split{split_id}"
    df = pd.read_parquet(run_dir / "predictions.parquet")
    valid = df[[SCORE, TARGET, DATE]].dropna()
    metrics = summary(valid[SCORE], valid[TARGET], valid[DATE], k=K)

    n_train = None
    stats_path = run_dir / "stats.json"
    if stats_path.exists():
        n_train = json.loads(stats_path.read_text(encoding="utf-8")).get("n_train")

    out = {
        "label": {"type": "fixed_horizon_fwd_r5"},
        "filter": "none",
        "n_train": n_train,
        "n_test": int(len(valid)),
        "metrics": metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return metrics, valid


def main():
    rows = []
    pooled = []
    for s in (1, 2, 3):
        m, valid = eval_one(s)
        rows.append({"split": s, "n_test": len(valid), **m})
        pooled.append(valid)

    # Pooled out-of-sample (concatenate the 3 disjoint test windows)
    allv = pd.concat(pooled, ignore_index=True)
    pm = summary(allv[SCORE], allv[TARGET], allv[DATE], k=K)
    rows.append({"split": "pooled", "n_test": len(allv), **pm})

    eval_dir = RESULTS / "tcn_wf_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(
        json.dumps({"label": {"type": "fixed_horizon_fwd_r5"}, "filter": "none",
                    "n_test": int(len(allv)), "metrics": pm}, indent=2),
        encoding="utf-8",
    )

    df = pd.DataFrame(rows)
    cols = ["split", "rank_ic_mean", "rank_ic_ir", "rank_ic_positive_rate",
            f"top{K}_return_mean", f"top{K}_return_sharpe", "n_dates", "n_test"]
    df = df[[c for c in cols if c in df.columns]]
    print("\n# TCN walk-forward evaluation (score=pump_ratio, target=fwd_r5)\n")
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nWrote per-split metrics.json + {eval_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
