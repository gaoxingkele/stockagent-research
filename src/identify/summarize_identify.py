"""SYN — synthesis of the leakage-free identification line.

Aggregates:
  - ID3 results/identify/ashare/stats.json   (identified LLM contribution)
  - WS2 results/identify/distill/stats.json  (LLM-weak-supervisor attribution)
  - DB2 results/identify/debias/finben_corrected.json (de-biased FinBen)
into results/identify/summary.md + paper/sections/figures/identify_summary.png.

Run: .venv-xpu\\Scripts\\python.exe -m src.identify.summarize_identify
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDENT = ROOT / "results/identify"
FIG_DIR = ROOT / "paper/sections/figures"


def summarize(ashare: dict, distill: dict, debias: dict) -> dict:
    rows = []
    for label in ("raw", "expert"):
        c = ashare["contribution"][label]
        rows.append(("ID3 LLM contribution (" + label + ")", c["mean"], c["lo"], c["hi"]))
    rows.append(("WS2 LLM-weak improvement",
                 distill["identified_improvement"],
                 distill["arm_B_llm_weak_refined"]["lo"] - distill["arm_A_true_labels"]["hi"],
                 distill["arm_B_llm_weak_refined"]["hi"] - distill["arm_A_true_labels"]["lo"]))
    return {"identified_estimates": rows,
            "leakage_validity_holds": ashare["leakage_validity"]["holds"],
            "finben_debiased": debias["corrected"]}


def to_md(s: dict) -> str:
    L = ["# Leakage-free identification of LLM contribution\n",
         f"Leakage validity (A-share) holds: **{s['leakage_validity_holds']}**\n",
         "## Identified estimates (date-clustered 95% CI)\n",
         "| Estimate | mean | lo | hi | spans 0? |", "|---|---|---|---|---|"]
    for name, m, lo, hi in s["identified_estimates"]:
        spans = "yes" if lo <= 0 <= hi else "no"
        L.append(f"| {name} | {m:+.3f} | {lo:+.3f} | {hi:+.3f} | {spans} |")
    L += ["\n## FinBen de-biased (reasoning-only) accuracy\n",
          "| Benchmark | debiased | memorization excess |", "|---|---|---|"]
    for bm, r in s["finben_debiased"].items():
        L.append(f"| {bm} | {r['debiased']:.3f} | {r['memorization_excess']:+.3f} |")
    return "\n".join(L) + "\n"


def run_real() -> dict:
    ashare = json.loads((IDENT / "ashare/stats.json").read_text(encoding="utf-8"))
    distill = json.loads((IDENT / "distill/stats.json").read_text(encoding="utf-8"))
    debias = json.loads((IDENT / "debias/finben_corrected.json").read_text(encoding="utf-8"))
    s = summarize(ashare, distill, debias)
    (IDENT / "summary.md").write_text(to_md(s), encoding="utf-8")
    _plot(s)
    return s


def _plot(s: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    names = [r[0] for r in s["identified_estimates"]]
    means = [r[1] for r in s["identified_estimates"]]
    los = [r[2] for r in s["identified_estimates"]]
    his = [r[3] for r in s["identified_estimates"]]
    y = np.arange(len(names))
    ax1.errorbar(means, y, xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                 fmt="o", color="#1f77b4", capsize=4)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("identified effect (clustered 95% CI)")
    ax1.set_title("A. LLM contribution on leakage-free A-shares\n(all ~0 / negative)", fontsize=10)

    bms = list(s["finben_debiased"].keys())
    full = [s["finben_debiased"][b]["debiased"] + s["finben_debiased"][b]["memorization_excess"] for b in bms]
    deb = [s["finben_debiased"][b]["debiased"] for b in bms]
    x = np.arange(len(bms)); w = 0.35
    ax2.bar(x - w / 2, [f * 100 for f in full], w, label="raw full-context", color="#1f77b4")
    ax2.bar(x + w / 2, [d * 100 for d in deb], w, label="de-biased (reasoning-only)", color="#d62728")
    ax2.axhline(50, color="gray", ls="--", lw=1); ax2.text(len(bms) - 0.5, 50.5, "chance", fontsize=8, color="gray", ha="right")
    ax2.set_xticks(x); ax2.set_xticklabels(bms); ax2.set_ylabel("accuracy (%)"); ax2.set_ylim(35, 85)
    ax2.set_title("B. FinBen de-biased: removing memorization\ncollapses 'skill' to ~chance", fontsize=10)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "identify_summary.png", dpi=150)


def main():
    print(json.dumps(run_real(), indent=2, default=str))


if __name__ == "__main__":
    main()
