"""NB6 — synthesis of the market-neutral / beta-vs-alpha line.

Reads results/identify/neutral (NB3) + results/identify/contrastive (NB5) and
writes results/identify/neutral_summary.md + a figure.

Run: .venv-xpu\\Scripts\\python.exe -m src.identify.summarize_neutral
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDENT = ROOT / "results/identify"
FIG_DIR = ROOT / "paper/sections/figures"


def summarize(neutral: dict, contrastive: dict) -> dict:
    sel = neutral.get("selection", {}).get("market_neutral", {})
    tim = neutral.get("timing", {}).get("market_neutral", {})
    out = {
        "leakage_validity_holds": neutral.get("leakage_validity", {}).get("holds"),
        "selection_market_neutral": {k: {"mean": v["mean"], "lo": v["lo"], "hi": v["hi"]}
                                     for k, v in sel.items()},
        "timing_market": {k: {"mean": v["mean"], "lo": v["lo"], "hi": v["hi"]}
                          for k, v in tim.items()},
        "long_short_baseline": neutral.get("long_short", {}).get("baseline", {}),
        "contrastive_arms": {a: {"rank_ic": contrastive["arms"][a]["market_neutral_rank_ic"],
                                 "long_short": contrastive["arms"][a]["long_short"]}
                             for a in contrastive.get("arms", {})},
    }
    return out


def to_md(s: dict) -> str:
    L = ["# Market-neutral identification: beta-timing vs alpha-selection\n",
         f"Leakage validity holds: **{s['leakage_validity_holds']}**\n",
         "## Identified LLM contribution (market-neutral, clustered 95% CI)\n",
         "| component | signal | mean | lo | hi | spans 0? |", "|---|---|---|---|---|---|"]
    for comp, key in (("selection (alpha)", "selection_market_neutral"), ("timing (beta)", "timing_market")):
        for sig, v in s[key].items():
            spans = "yes" if v["lo"] <= 0 <= v["hi"] else "no"
            L.append(f"| {comp} | {sig} | {v['mean']:+.3f} | {v['lo']:+.3f} | {v['hi']:+.3f} | {spans} |")
    L += ["\n## Tradable market-neutral long-short (annualized Sharpe; single held-out window -- not alpha evidence)\n",
          "| source | Sharpe | mean/period | n_dates |", "|---|---|---|---|"]
    b = s.get("long_short_baseline", {})
    if b:
        L.append(f"| NB3 baseline (full window) | {b.get('annualized_sharpe',float('nan')):.2f} | {b.get('ls_mean_per_period',float('nan')):+.4f} | {b.get('n_dates')} |")
    for a, v in s.get("contrastive_arms", {}).items():
        ls = v["long_short"]
        L.append(f"| NB5 {a} | {ls.get('annualized_sharpe',float('nan')):.2f} | {ls.get('ls_mean_per_period',float('nan')):+.4f} | {ls.get('n_dates')} |")
    return "\n".join(L) + "\n"


def _plot(s: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = []
    for sig, v in s["selection_market_neutral"].items():
        rows.append((f"selection/{sig}", v))
    for sig, v in s["timing_market"].items():
        rows.append((f"timing/{sig}", v))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(rows))
    means = [r[1]["mean"] for r in rows]
    lo = [r[1]["mean"] - r[1]["lo"] for r in rows]; hi = [r[1]["hi"] - r[1]["mean"] for r in rows]
    ax.errorbar(means, y, xerr=[lo, hi], fmt="o", color="#1f77b4", capsize=4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("identified LLM contribution (market-neutral, 95% CI)")
    ax.set_title("Beta-timing vs alpha-selection: LLM contribution ~0 on leakage-free A-shares", fontsize=9)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "market_neutral_summary.png", dpi=150)


def run_real() -> dict:
    neutral = json.loads((IDENT / "neutral/stats.json").read_text(encoding="utf-8"))
    contrastive = json.loads((IDENT / "contrastive/stats.json").read_text(encoding="utf-8"))
    s = summarize(neutral, contrastive)
    (IDENT / "neutral_summary.md").write_text(to_md(s), encoding="utf-8")
    _plot(s)
    return s


def main():
    print(json.dumps(run_real(), indent=2, default=str))


if __name__ == "__main__":
    main()
