"""Aggregate run results into paper Table 1 / Table 2 format.

Scans results/*/metrics.json and produces a markdown comparison table.

Usage:
    python -m src.evaluation.aggregate                 # all runs
    python -m src.evaluation.aggregate --filter e1     # only e1.x runs
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def load_runs(name_filter: str | None = None) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(RESULTS.iterdir()):
        if not run_dir.is_dir():
            continue
        if name_filter and name_filter not in run_dir.name:
            continue
        meta = run_dir / "metrics.json"
        if not meta.exists():
            continue
        with open(meta, encoding="utf-8") as f:
            d = json.load(f)
        row = {
            "run": run_dir.name,
            "label": d.get("label", {}).get("type", "?"),
            "filter": d.get("filter", "none"),
            "n_train": d.get("n_train"),
            "n_test": d.get("n_test"),
            **d.get("metrics", {}),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no runs found)\n"
    cols = [
        "run", "label", "filter",
        "rank_ic_mean", "rank_ic_ir", "rank_ic_positive_rate",
        "top20_return_mean", "top20_return_sharpe",
        "n_train", "n_test",
    ]
    cols = [c for c in cols if c in df.columns]
    fmt = df[cols].copy()

    # Number formatting
    fmt_map = {
        "rank_ic_mean": "{:+.4f}",
        "rank_ic_ir": "{:+.3f}",
        "rank_ic_positive_rate": "{:.1%}",
        "top20_return_mean": "{:.4f}",
        "top20_return_sharpe": "{:.2f}",
        "n_train": "{:,}",
        "n_test": "{:,}",
    }
    for c, f in fmt_map.items():
        if c in fmt.columns:
            fmt[c] = fmt[c].apply(lambda x: f.format(x) if pd.notna(x) else "—")

    return fmt.to_markdown(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default=None, help="Substring match on run name")
    ap.add_argument("--out", default=None, help="Output markdown path (default: stdout)")
    args = ap.parse_args()

    df = load_runs(args.filter)
    table = format_table(df)

    header = "# Experiment Aggregate\n\n"
    body = header + table + "\n"

    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(body)


if __name__ == "__main__":
    main()
