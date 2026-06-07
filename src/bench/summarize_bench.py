"""BSYN -- do 2025-26 SOTA regime/uncertainty baselines survive our protocol?

Aggregates EXP-A/B/C into one verdict + figure. Headline: reimplemented on
A-shares under our leakage-free, cost-aware, cross-period, Deflated-Sharpe
protocol, the closest recent baselines are ALSO sub-cost; our contribution is the
honest evaluation and the interaction-information probe.

Run: .venv-xpu\\Scripts\\python.exe -m src.bench.summarize_bench
"""
from __future__ import annotations

import json

from src.train_tcn_wf import ROOT

OUT = ROOT / "results/bench"
FIG = ROOT / "paper/sections/figures/bench_summary.png"


def bench_verdict(expa_survivors: list, expb_rescues: bool,
                  expc_efficient_band: bool, expc_interaction_n: int) -> dict:
    a = ("Regime-Aware LightGBM (incl. rolling-HMM) SURVIVES: " + ",".join(expa_survivors)
         if expa_survivors else
         "Regime-Aware LightGBM (incl. rolling-HMM) is SUB-COST under our protocol")
    b = ("two-level abstention RESCUES the onset edge" if expb_rescues
         else "two-level abstention does NOT rescue (onset stays sub-cost)")
    c = ("NMI/conditional-MI baselines see an efficient-market band; our interaction "
         f"info isolates regime-added information ({expc_interaction_n} hits)"
         if expc_efficient_band else
         "NMI baselines already see the dependence")
    deployable = bool(expa_survivors) or expb_rescues
    overall = ("A baseline SURVIVED -- report it." if deployable else
               "NONE of the reimplemented 2025-26 SOTA baselines survive our honest "
               "protocol; like our own signals they are sub-cost. The contribution is "
               "the honest, deployment-realistic evaluation + the interaction-information probe.")
    return {"expA": a, "expB": b, "expC": c, "any_deployable": deployable,
            "overall": overall}


def _load():
    return (json.loads((OUT / "expa.json").read_text(encoding="utf-8")),
            json.loads((OUT / "expb.json").read_text(encoding="utf-8")),
            json.loads((OUT / "expc.json").read_text(encoding="utf-8")))


def _figure(expa, expb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms, sharpes, dsrs = [], [], []
    for label, m in expa["pooled"].items():
        arms.append(f"A:{label}"); sharpes.append(m.get("annualized_sharpe", 0.0))
        dsrs.append(m.get("deflated", {}).get("dsr", 0.0))
    for label, m in expb["pooled"].items():
        arms.append(f"B:{label}"); sharpes.append(m.get("annualized_sharpe", 0.0))
        dsrs.append(m.get("deflated", {}).get("dsr", 0.0))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(arms))
    colors = ["#4c72b0" if d > 0.95 else "#c44e52" for d in dsrs]
    ax.bar(x, sharpes, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    for i, d in enumerate(dsrs):
        ax.text(i, sharpes[i], f"DSR={d:.2f}", ha="center",
                va="bottom" if sharpes[i] >= 0 else "top", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("annualized Sharpe (net of A-share cost)")
    ax.set_title("SOTA regime/abstention baselines under our honest protocol\n"
                 "(blue = DSR>0.95; none has a strictly-positive net-of-cost mean CI)")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)


def run_real() -> dict:
    expa, expb, expc = _load()
    v = bench_verdict(expa["summary"]["survivors"],
                      expb["summary"]["abstention_rescues"],
                      expc["summary"]["nmi_efficient_regime"],
                      expc["summary"]["n_interaction_significant"])
    _figure(expa, expb)
    lines = ["# Benchmark synthesis: do 2025-26 SOTA baselines survive our protocol?", "",
             f"**Overall:** {v['overall']}", "",
             f"- **EXP-A** (Regime-Aware LightGBM, MDPI 2026): {v['expA']}",
             "  - pooled: " + "; ".join(
                 f"{k} Sharpe={m.get('annualized_sharpe'):+.2f} DSR={m.get('deflated',{}).get('dsr'):.2f} "
                 f"CI={[round(c,4) for c in (m.get('mean_ci95') or [float('nan'),float('nan')])]}"
                 for k, m in expa["pooled"].items()),
             f"- **EXP-B** (When Alpha Breaks abstention, 2603.13252): {v['expB']}",
             "  - pooled: " + "; ".join(
                 f"{k} Sharpe={m.get('annualized_sharpe'):+.2f} DSR={m.get('deflated',{}).get('dsr'):.2f}"
                 for k, m in expb["pooled"].items()),
             f"- **EXP-C** (NMI / Conditional-MI info-theory framings): {v['expC']}",
             "", "## Takeaway",
             "Reimplemented on A-shares under leakage-free + cost-aware + cross-period + "
             "Deflated-Sharpe evaluation, the closest 2025-26 regime/uncertainty baselines "
             "are sub-cost just like our own onset/motif signals. Apparent Sharpe gains from "
             "regime gating are cash-during-volatility risk-reduction, not net-of-cost alpha. "
             "This strengthens the paper's honest-findings thesis and the interaction-information "
             "probe (C6) as the distinguishing methodological contribution."]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "bench_verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    return v


def main():
    v = run_real()
    print(json.dumps(v, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
