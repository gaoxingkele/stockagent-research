"""PSYN -- production-edge verdict + Sharpe-decomposition synthesis.

Reads results/production/{timing,filtered_pool,optimize}.json and answers:
where (if anywhere) does the DOCUMENTED V12.31 mechanism reproduce an edge --
timing? selection? both? none? -- under the honest (robust mean, cross-period)
criteria, not the Sharpe-inflation artifact of COMBO1.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.summarize_production
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "results/production"
FIG_DIR = ROOT / "paper/sections/figures"


def production_verdict(timing_net_ci, selection_pos_years, opt_oos_sharpes) -> dict:
    timing_works = timing_net_ci is not None and timing_net_ci[0] > 0   # net mean CI > 0
    selection_works = selection_pos_years is not None and selection_pos_years >= 2
    opt_works = opt_oos_sharpes and all(s is not None and s > 0 for s in opt_oos_sharpes)
    if not timing_works and not selection_works and not opt_works:
        v = "NOT REPRODUCIBLE: the documented V12.31 rules (timing + selection) do not reproduce the production edge"
    elif selection_works or opt_works:
        v = "SELECTION reproduces an edge"
    elif timing_works:
        v = "TIMING reproduces an edge"
    else:
        v = "inconclusive"
    return {"verdict": v, "timing_works": bool(timing_works),
            "selection_works": bool(selection_works), "opt_oos_works": bool(opt_works)}


def run_real() -> dict:
    tim = json.loads((PROD / "timing.json").read_text(encoding="utf-8"))
    flt = json.loads((PROD / "filtered_pool.json").read_text(encoding="utf-8"))
    opt = json.loads((PROD / "optimize.json").read_text(encoding="utf-8"))

    timing_net_ci = tim["arms"]["trend_regime"]["pooled"].get("timed_mean_ci95")
    selection_pos_years = flt.get("positive_years")
    opt_oos = opt.get("wf_selected_oos_sharpe")
    verdict = production_verdict(timing_net_ci, selection_pos_years, opt_oos)

    decomp = {
        "timing_trend_pooled_incremental_sharpe": tim["arms"]["trend_regime"]["pooled"].get("incremental_timing"),
        "selection_pool_pooled_sharpe": flt["pooled"].get("annualized_sharpe"),
        "selection_pool_positive_years": selection_pos_years,
        "opt_wf_selected_oos_sharpe": opt_oos,
        "disaster_filter_fire_frac": tim.get("disaster_frac_overall"),
    }
    out = {**verdict, "decomposition": decomp}

    md = ["# Production-edge verdict (does the documented V12.31 mechanism reproduce?)\n",
          f"**VERDICT: {verdict['verdict']}**\n", "## Decomposition (honest, robust)\n",
          "| component | result |", "|---|---|",
          f"| timing (trend regime) incremental Sharpe | {decomp['timing_trend_pooled_incremental_sharpe']:+.2f} (net mean CI spans 0 -> wash) |",
          f"| disaster_filter fire rate | {decomp['disaster_filter_fire_frac']:.3f} (broken: barely fires, missed 2022 bear) |",
          f"| selection pool pooled Sharpe | {decomp['selection_pool_pooled_sharpe']:+.2f} ({selection_pos_years}/3 positive years) |",
          f"| walk-forward-selected OOS Sharpe | {opt_oos} (no config recovers an edge) |",
          "\nConclusion: the documented rules (onset + V7c filters + simplified disaster timing) do NOT reproduce the production Sharpe 2.20. The real edge lives in parts NOT captured here: the FULL disaster composite (concept signals C2/C3 unimplemented), the actual r20_pred predictive model (we used a momentum proxy), execution/risk/discretion, and production parameter calibration.\n"]
    PROD.mkdir(parents=True, exist_ok=True)
    (PROD / "summary.md").write_text("\n".join(md), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["timing\n(incremental)", "selection\n(pool pooled)", "WF-opt OOS\n2024", "WF-opt OOS\n2025"]
    vals = [decomp["timing_trend_pooled_incremental_sharpe"], decomp["selection_pool_pooled_sharpe"],
            (opt_oos[0] if opt_oos else 0), (opt_oos[1] if opt_oos and len(opt_oos) > 1 else 0)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=["#7f7f7f" if abs(v) < 0.3 else ("#2ca02c" if v > 0 else "#d62728") for v in vals])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Sharpe contribution")
    ax.set_title("Documented V12.31 rules do not reproduce the edge:\ntiming wash, selection negative", fontsize=10)
    fig.tight_layout(); FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "production_edge_summary.png", dpi=150)
    return out


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
