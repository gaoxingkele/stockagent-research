"""E3 — temporal-leakage analysis on ACL18 (FLARE stock-movement).

Consolidates the evidence that a modern LLM's "stock-movement prediction" skill
on a pre-training-cutoff benchmark is memorization / temporal leakage, not
forecasting:

  1. LLM accuracy with the FULL benchmark context (price history + tweets);
  2. LLM accuracy with NO context (ticker + target date only) -- if this is as
     high or higher, the model is recalling memorized history, not analysing;
  3. mega-cap vs the rest (famous tickers are better memorized);
  4. all vs the tabular price baseline (near chance).

Compares against the directional accuracy a genuine next-day predictor should
reach (~50-53%). Writes results/e3_acl/leakage_summary.json + a figure.

Usage:
    .venv-xpu\\Scripts\\python.exe -m src.eval_e3.analyze_leakage --dataset acl
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/e3_flare"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper/sections/figures"

MEGA = {"aapl", "amzn", "msft", "googl", "goog", "fb", "csco", "intc",
        "ibm", "orcl", "qcom", "nvda", "amd", "nflx", "tsla"}


def llm_acc(df: pd.DataFrame) -> tuple[float, int]:
    m = df[df["llm_direction"].notna()].copy()
    m["pred"] = (m["llm_direction"] == "Fall").astype(int)   # gold=1 <-> Fall
    return float((m["pred"] == m["gold"]).mean()), int(len(m))


def boot_acc_ci(df, n_boot=2000, seed=42):
    """Date-clustered bootstrap CI of LLM accuracy."""
    m = df[df["llm_direction"].notna()].copy()
    m["correct"] = ((m["llm_direction"] == "Fall").astype(int) == m["gold"]).astype(int)
    rng = np.random.default_rng(seed)
    dates = m["date"].values
    uniq = np.unique(dates)
    by_date = {d: np.where(dates == d)[0] for d in uniq}
    correct = m["correct"].values
    accs = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_date[d] for d in draw])
        accs.append(correct[idx].mean())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="acl")
    args = ap.parse_args()
    out_dir = RESULTS / f"e3_{args.dataset}"

    te = pd.read_parquet(BASE / args.dataset / "parsed_test.parquet")[["id", "gold", "ticker", "date"]]
    full = te.merge(pd.read_parquet(out_dir / "llm_test.parquet")[["id", "llm_direction"]], on="id")
    noc = te.merge(pd.read_parquet(out_dir / "llm_test_nocontext.parquet")[["id", "llm_direction"]], on="id")
    base = pd.read_parquet(out_dir / "baseline_test.parquet")
    base_acc = float(((base["baseline_p"] >= 0.5).astype(int) == base["gold"]).mean())

    a_full, n_full = llm_acc(full)
    a_noc, n_noc = llm_acc(noc)
    ci_full = boot_acc_ci(full)
    ci_noc = boot_acc_ci(noc)

    # mega-cap breakdown (full context)
    full["mega"] = full["ticker"].isin(MEGA)
    mega_acc, _ = llm_acc(full[full["mega"]])
    rest_acc, _ = llm_acc(full[~full["mega"]])

    summary = {
        "dataset": args.dataset,
        "n_test": int(len(te)),
        "baseline_acc": base_acc,
        "llm_full_context": {"acc": a_full, "n": n_full, "ci95": ci_full},
        "llm_no_context": {"acc": a_noc, "n": n_noc, "ci95": ci_noc},
        "context_contribution_pp": (a_full - a_noc) * 100,
        "mega_cap_acc": mega_acc,
        "non_mega_acc": rest_acc,
        "plausible_skill_ceiling": 0.53,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leakage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== E3 temporal-leakage / {args.dataset} (n={len(te)}) ===")
    print(f"  tabular price baseline      acc={base_acc:.4f}")
    print(f"  LLM, FULL context           acc={a_full:.4f}  95%CI [{ci_full[0]:.3f},{ci_full[1]:.3f}]  (n={n_full})")
    print(f"  LLM, NO context (tick+date) acc={a_noc:.4f}  95%CI [{ci_noc[0]:.3f},{ci_noc[1]:.3f}]  (n={n_noc})")
    print(f"  -> context contributes {(a_full-a_noc)*100:+.2f}pp (negative = context HURTS recall)")
    print(f"  mega-cap acc={mega_acc:.4f}  vs  non-mega acc={rest_acc:.4f}")
    print(f"  plausible next-day skill ceiling ~0.53")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4))
    labels = ["Tabular\nbaseline", "LLM\nfull context", "LLM\nno context\n(ticker+date)"]
    vals = [base_acc * 100, a_full * 100, a_noc * 100]
    errs = [[0, (a_full - ci_full[0]) * 100, (a_noc - ci_noc[0]) * 100],
            [0, (ci_full[1] - a_full) * 100, (ci_noc[1] - a_noc) * 100]]
    bars = ax.bar(labels, vals, yerr=errs, capsize=5,
                  color=["#2ca02c", "#1f77b4", "#d62728"])
    ax.axhline(53, color="gray", ls="--", lw=1)
    ax.text(2.4, 53.6, "plausible next-day\nskill ceiling ~53%", fontsize=8, color="gray", ha="right")
    ax.axhline(50, color="k", lw=0.6)
    ax.set_ylabel("directional accuracy (%)")
    ax.set_ylim(45, 80)
    ax.set_title("ACL18: removing all input RAISES LLM accuracy to 76%\n(temporal-leakage / memorization, not forecasting)", fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "e3_leakage.png", dpi=150)
    print(f"\nWrote {out_dir/'leakage_summary.json'} and {FIG_DIR/'e3_leakage.png'}")


if __name__ == "__main__":
    main()
