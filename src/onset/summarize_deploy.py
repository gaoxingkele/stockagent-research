"""DSYN -- deployability verdict + synthesis.

Reads results/deploy/{long_only,robustness}.json and applies SIGN-D1:
DEPLOYABLE only if the LONG-ONLY net market-excess (1) pooled CI excludes 0,
(2) is positive in >=2/3 of the 2025 splits, AND (3) holds across >=1 extra
year (2023/2024). Otherwise 'statistically-real-but-not-deployable (collapsed
cross-period)' or 'collapsed'.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.summarize_deploy
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEP = ROOT / "results/deploy"
FIG_DIR = ROOT / "paper/sections/figures"


def deploy_verdict(pooled_net_ci, net_positive_splits, holds_across_years) -> str:
    pooled_sig = pooled_net_ci is not None and pooled_net_ci[0] > 0
    gate_2025 = pooled_sig and net_positive_splits >= 2
    if gate_2025 and holds_across_years:
        return "DEPLOYABLE: long-only net-excess significant on 2025 AND holds cross-period"
    if gate_2025 and not holds_across_years:
        return "STATISTICALLY-REAL-ON-2025-BUT-NOT-DEPLOYABLE (edge collapses cross-period)"
    if pooled_sig:
        return "WEAK: pooled significant but not robust per-split"
    return "COLLAPSED / null"


def run_real() -> dict:
    lo = json.loads((DEP / "long_only.json").read_text(encoding="utf-8"))
    rob = json.loads((DEP / "robustness.json").read_text(encoding="utf-8"))
    pooled_ci = lo["pooled"]["net"].get("mean_ci95")
    verdict = deploy_verdict(pooled_ci, lo.get("net_positive_splits", 0), rob.get("holds_across_years", False))

    sharpes = {"2025 pooled (LO2)": lo["pooled"]["net"]["annualized_sharpe"],
               "2023 (ROB1)": rob["per_year_net"]["2023"]["annualized_sharpe"],
               "2024 (ROB1)": rob["per_year_net"]["2024"]["annualized_sharpe"]}
    out = {"verdict": verdict, "long_only_net_sharpe_by_period": sharpes,
           "pooled_2025_net_ci": pooled_ci, "holds_across_years": rob.get("holds_across_years")}

    md = ["# Deployability verdict (long-only, net of realistic A-share cost)\n",
          f"**VERDICT: {verdict}**\n", "## Long-only net annualized Sharpe by period\n",
          "| period | net Sharpe |", "|---|---|"]
    for k, v in sharpes.items():
        md.append(f"| {k} | {v:+.2f} |")
    md.append(f"\nholds across years: **{rob.get('holds_across_years')}**\n")
    DEP.mkdir(parents=True, exist_ok=True)
    (DEP / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ks = list(sharpes); vs = [sharpes[k] for k in ks]
    ax.bar(ks, vs, color=["#2ca02c" if v > 0 else "#d62728" for v in vs])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("long-only net annualized Sharpe")
    ax.set_title("alpha1 collapses cross-period: 2025-only, negative in 2023/2024", fontsize=10)
    fig.tight_layout(); FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "deploy_summary.png", dpi=150)
    return out


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
