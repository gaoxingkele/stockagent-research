"""Methods-compared overview tree for the paper.

Renders every method / arm compared across the project's 11 lines, colour-coded by
verdict. Output: paper/sections/figures/methods_tree.png.

Run: .venv-xpu\\Scripts\\python.exe -m src.figviz.methods_tree
"""
from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from src.train_tcn_wf import ROOT

FIG = ROOT / "paper/sections/figures/methods_tree.png"

RED, GREEN, AMBER, GRAY = "#c44e52", "#55a868", "#dd8452", "#8c8c8c"

LINES = [
    (1, "Main framework — walk-forward (§5)",
     "4 baselines: LGBM · LLM-raw (Sonnet) · LLM-expert · expert-rule  |  routers A–I (9)  |  "
     "Oracle bound  |  Pattern Core: LGBM → TCN+Cross-Attn → Barlow-Twins SSL  |  Sonnet vs Haiku",
     "Hybrid ≈ LGBM (within noise)", AMBER),
    (2, "FinBen temporal leakage (C5 precursor)",
     "ACL18 / BigData22 / CIKM18 × {full-context, no-context}  |  mega-cap gradient  |  A-share control",
     "US 'skill' = memorization", RED),
    (3, "Leakage-free identification (C5)",
     "ID3: LLM contribution raw vs expert  |  WS2: true-labels vs LLM-weak-supervised  |  "
     "DB2: FinBen de-biased vs chance",
     "LLM contribution ≈ 0; weak-sup harmful", RED),
    (4, "Leakage frontier",
     "multi-model leakage probe  |  cross-market de-bias",
     "Shelved (novelty check failed)", GRAY),
    (5, "Market-neutral line",
     "NB3: selection vs timing (raw / expert)  |  NB5: contrastive encoder vs raw",
     "No idiosyncratic alpha; weak timing hint", RED),
    (6, "Candlestick-onset (alpha1 hunt)",
     "K3: candle-geometry LGBM  |  K4: candle-sequence GRU  |  K5: factors / candle / factors+candle",
     "factors+candle passes SIGN-K1 (in-sample 2025)", AMBER),
    (7, "Deployability gauntlet",
     "LO2: long-only top-K excess  |  ROB1: cross-period 2023/24 + liquidity + top-K  |  CAP1: capacity",
     "Collapses cross-period → NOT deployable", RED),
    (8, "Production-edge (what V12.31 earns from)",
     "TIM2: disaster vs trend timing  |  FILT2: extreme-filter pool  |  COMBO1  |  OPT1 knob sweep",
     "Not reproducible (timing wash, selection < 0)", RED),
    (9, "Onset-motif — information-theoretic (C6)",
     "MI2: marginal vs conditional vs INTERACTION info (Z-label perm) × {trend, vol, disaster}  |  "
     "MI3: per-year 2022–25 stability",
     "Regime adds STABLE conditional info (52/78)", GREEN),
    (10, "Motif tradability",
     "TRD1: directionality / monotonicity / variance-vs-mean  |  TRD2: regime-gated long-only",
     "Information-only — sub-cost", RED),
    (11, "SOTA comparison (§6.6)",
     "EXP-A: regime LGBM (plain / HMM / trend)  |  EXP-B: two-level abstention  |  "
     "EXP-C: NMI vs interaction info",
     "SOTA also sub-cost under our protocol", RED),
]


def render():
    n = len(LINES)
    row_h = 1.0
    fig_h = 2.0 + n * row_h * 1.15
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, n * 1.15 + 2.0)
    ax.axis("off")

    top = n * 1.15 + 1.2
    # root
    ax.text(2, top + 0.45, "stockagent-research  —  A-share movement-onset detection",
            fontsize=16, fontweight="bold", va="center")
    ax.text(2, top - 0.05, "60+ methods / arms compared across 11 lines  "
            "(green = robust positive · amber = in-sample / within-noise · red = null / sub-cost / not-deployable · gray = shelved)",
            fontsize=10, color="#444444", va="center")

    trunk_x = 3.0
    ys = []
    for i, (num, title, methods, verdict, color) in enumerate(LINES):
        y = top - 1.2 - i * 1.15
        ys.append(y)
        # connector tick from trunk
        ax.plot([trunk_x, 6.0], [y, y], color="#bbbbbb", lw=1.0, zorder=1)
        # verdict colour chip
        ax.add_patch(FancyBboxPatch((6.2, y - 0.42), 1.1, 0.84,
                     boxstyle="round,pad=0.02", linewidth=0, facecolor=color, zorder=3))
        # branch box
        ax.add_patch(FancyBboxPatch((7.6, y - 0.46), 90.0, 0.92,
                     boxstyle="round,pad=0.02", linewidth=0.8,
                     edgecolor="#dddddd", facecolor="#fafafa", zorder=2))
        ax.text(8.2, y + 0.22, f"{num}. {title}", fontsize=12, fontweight="bold",
                va="center", zorder=4)
        wrapped = textwrap.fill(methods, width=118)
        ax.text(8.2, y - 0.20, wrapped, fontsize=8.4, color="#333333",
                va="center", zorder=4)
        ax.text(96.8, y + 0.22, verdict, fontsize=9.5, fontweight="bold",
                color=color, ha="right", va="center", zorder=4)
    # trunk
    ax.plot([trunk_x, trunk_x], [ys[-1], ys[0]], color="#999999", lw=2.0, zorder=1)

    legend = [Patch(facecolor=GREEN, label="robust positive"),
              Patch(facecolor=AMBER, label="in-sample / within-noise"),
              Patch(facecolor=RED, label="null / sub-cost / not-deployable"),
              Patch(facecolor=GRAY, label="shelved")]
    ax.legend(handles=legend, loc="lower center", ncol=4, fontsize=9,
              frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return FIG


if __name__ == "__main__":
    print("wrote", render())
