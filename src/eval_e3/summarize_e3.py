"""E3 — combine the per-dataset temporal-leakage summaries (ACL18 / BigData22 /
CIKM18) into one cross-benchmark table + grouped bar figure for the paper.

Run analyze_leakage.py for each dataset first (it writes leakage_summary.json).

Usage:
    .venv-xpu\\Scripts\\python.exe -m src.eval_e3.summarize_e3
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper/sections/figures"

DATASETS = [("acl", "ACL18\n(2014-16)"), ("bigdata", "BigData22\n(2019-21)"), ("cikm", "CIKM18\n(2017)")]


def main():
    rows = []
    for key, _ in DATASETS:
        p = RESULTS / f"e3_{key}" / "leakage_summary.json"
        if not p.exists():
            print(f"  MISSING {p} -- run analyze_leakage --dataset {key} first")
            continue
        rows.append((key, json.loads(p.read_text(encoding="utf-8"))))

    if not rows:
        raise SystemExit("no summaries found")

    print(f"\n{'dataset':<10} {'n':>5} {'baseline':>9} {'LLM full':>9} {'LLM no-ctx':>11} {'ctx Δpp':>8} {'mega':>7} {'non-mega':>9}")
    table_md = ["| Dataset | n | Baseline | LLM full ctx | LLM no ctx | Δ ctx (pp) | mega-cap | non-mega |",
                "|---|---|---|---|---|---|---|---|"]
    for key, s in rows:
        f = s["llm_full_context"]["acc"]; nc = s["llm_no_context"]["acc"]
        print(f"{key:<10} {s['n_test']:>5} {s['baseline_acc']*100:>8.1f}% "
              f"{f*100:>8.1f}% {nc*100:>10.1f}% {s['context_contribution_pp']:>+7.2f} "
              f"{s['mega_cap_acc']*100:>6.1f}% {s['non_mega_acc']*100:>8.1f}%")
        table_md.append(f"| {key} | {s['n_test']} | {s['baseline_acc']*100:.1f}% | "
                        f"{f*100:.1f}% | {nc*100:.1f}% | {s['context_contribution_pp']:+.2f} | "
                        f"{s['mega_cap_acc']*100:.1f}% | {s['non_mega_acc']*100:.1f}% |")

    out = RESULTS / "e3_summary.md"
    out.write_text("# E3 cross-benchmark temporal leakage\n\n" + "\n".join(table_md) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")

    # grouped bar figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [lbl for _, lbl in DATASETS if (RESULTS / f"e3_{_}" / "leakage_summary.json").exists()]
    keys = [k for k, _ in DATASETS if (RESULTS / f"e3_{k}" / "leakage_summary.json").exists()]
    summ = {k: json.loads((RESULTS / f"e3_{k}" / "leakage_summary.json").read_text(encoding="utf-8")) for k in keys}
    x = np.arange(len(keys)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w, [summ[k]["baseline_acc"]*100 for k in keys], w, label="tabular baseline", color="#2ca02c")
    ax.bar(x,     [summ[k]["llm_full_context"]["acc"]*100 for k in keys], w, label="LLM full context", color="#1f77b4")
    ax.bar(x + w, [summ[k]["llm_no_context"]["acc"]*100 for k in keys], w, label="LLM no context (ticker+date)", color="#d62728")
    ax.axhline(53, color="gray", ls="--", lw=1); ax.text(len(keys)-0.5, 53.5, "~53% skill ceiling", fontsize=8, color="gray", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("directional accuracy (%)"); ax.set_ylim(45, 80)
    ax.set_title("Temporal leakage across the FinBen stock-movement suite\n(LLM no-context >= full-context on every benchmark)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "e3_leakage_suite.png", dpi=150)
    print(f"Wrote {FIG_DIR / 'e3_leakage_suite.png'}")


if __name__ == "__main__":
    main()
