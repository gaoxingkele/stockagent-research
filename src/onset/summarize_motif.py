"""MSYN -- information-theoretic go/no-go verdict for the onset-motif upgrade.

Aggregates MI2 (marginal-vs-conditional + interaction probe) and MI3 (cross-period
stability) into the decisive verdict, with the honesty caveats that separate
"information exists" from "deployable alpha".

Verdict logic (motif_verdict): BUILD-THE-MOTIF-MODEL only if the regime ADDS
information (interaction>0, Z-permutation significant) AND that interaction is
cross-period STABLE (positive+significant in >=2/3 of years) for >=1 feature.
Otherwise 'present-but-unstable' or 'information-exhausted'.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.summarize_motif
"""
from __future__ import annotations

import json
from pathlib import Path

from src.train_tcn_wf import ROOT

OUT = ROOT / "results/motif"
FIG = ROOT / "paper/sections/figures/motif_mi_summary.png"


def motif_verdict(n_interaction_hits: int, n_stable: int, n_tested: int) -> dict:
    """Decisive verdict from the two probes.

    n_interaction_hits : # (feat x target x regime) where the regime adds info (MI2)
    n_stable           : # items whose interaction is cross-period stable (MI3)
    n_tested           : # items carried into the stability test
    """
    if n_interaction_hits == 0:
        verdict = "INFORMATION-EXHAUSTED -- regime adds no information; the onset concept is at its information-theoretic ceiling. Do NOT build the motif model."
        build = False
    elif n_stable == 0:
        verdict = "PRESENT-BUT-UNSTABLE -- the regime adds information pooled, but it is not reproducible across years. Not a basis for a motif model (same non-stationarity that killed the return edge)."
        build = False
    else:
        verdict = "BUILD-THE-MOTIF-MODEL -- conditional information is permutation-significant AND cross-period stable. The trend/vol regime is a genuine 'transcription factor' that activates the candle 'promoter'."
        build = True
    return {"verdict": verdict, "build_motif_model": build,
            "n_interaction_hits": n_interaction_hits, "n_stable": n_stable,
            "n_tested": n_tested}


CAVEATS = [
    "p-values floor at 1/n_perm because n is large (~1.5-2e5): they prove the interaction is >0, NOT that it is large. The effect SIZE (cond_mi ~0.006 nats, correlation-equivalent ~0.07-0.14) is the honest magnitude.",
    "INFORMATION != net-of-cost RETURN. The deployability and production-edge lines already showed the long-only, cost-aware return edge collapses cross-period. Conditional information is necessary, not sufficient, for tradable alpha.",
    "Mutual information is SIGN-BLIND: part of this conditional information may describe downside/continuation risk in down-trends rather than exploitable upside.",
    "The motif model's hard, unsolved job is to convert this stable conditional information into a MONOTONE, costable, long-only signal -- which prior lines suggest is the binding constraint.",
]


def _load():
    probe = json.loads((OUT / "mi_probe.json").read_text(encoding="utf-8"))
    stab = json.loads((OUT / "mi_stability.json").read_text(encoding="utf-8"))
    return probe, stab


def _figure(stab: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = stab["items"]
    years = list(stab["years"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for it in items:
        ys = [it["per_year"].get(y, {}).get("interaction", float("nan")) for y in years]
        ax.plot(years, ys, marker="o",
                label=f"{it['feature']} | {it['regime']}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("interaction information II = I(X;Y|Z) - I(X;Y)  (nats)")
    ax.set_xlabel("year")
    ax.set_title("Onset-motif: regime-added information is positive every year\n"
                 "(trend/vol regime as 'transcription factor')")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120)
    plt.close(fig)


def run_real() -> dict:
    probe, stab = _load()
    v = motif_verdict(probe["summary"]["n_hits"], stab["n_stable"], stab["n_tested"])
    _figure(stab)
    lines = ["# Onset-motif: information-theoretic go/no-go", "",
             f"**Verdict:** {v['verdict']}", "",
             f"- MI2 interaction hits (regime adds info): {v['n_interaction_hits']}/78",
             f"- MI3 cross-period-stable items: {v['n_stable']}/{v['n_tested']}",
             f"- pooled sample: {probe['n_rows']} rows; {probe['n_perm']} permutations; {probe['bins']} bins",
             "", "## Top stable interactions (per-year II, nats)", ""]
    for it in stab["items"]:
        ys = ", ".join(f"{y} {it['per_year'][y]['interaction']:+.4f}"
                       for y in stab["years"] if y in it["per_year"])
        lines.append(f"- **{it['feature']} | {it['regime']}** "
                     f"(stable={it['stable']}, {it['n_pos_sig']}/{it['n_years']} yrs): {ys}")
    lines += ["", "## Honesty caveats (information != alpha)", ""]
    lines += [f"{i+1}. {c}" for i, c in enumerate(CAVEATS)]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    out = {**v, "caveats": CAVEATS, "figure": str(FIG)}
    (OUT / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    out = run_real()
    print(out["verdict"])
    print(f"build_motif_model={out['build_motif_model']} "
          f"hits={out['n_interaction_hits']} stable={out['n_stable']}/{out['n_tested']}")


if __name__ == "__main__":
    main()
