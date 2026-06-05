"""MI3 -- REAL ($0): cross-period stability of the interaction information.

MI2 found that the TREND regime adds information (interaction II>0) for several
candle features -- POOLED across 2022-2025. But pooled statistical dependence at
n=2e5 is necessary, not sufficient: if the interaction only lives in one year it
is not a basis for a motif model (the same non-stationarity that killed the
return edge, SIGN-D1, applies to INFORMATION).

This recomputes the interaction test PER YEAR for the regime + features that won
in MI2, and reports for each (feature, regime) the per-year interaction nats and
Z-permutation p-value, plus how many years it is positive-and-significant. A
motif is "stable" only if it clears the bar in >=2/3 of the available years.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_mi_stability
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_tcn_wf import ROOT
from src.onset.mutual_info import interaction_pvalue
from src.onset.run_mi_probe import load_panel, build, OUT, N_PERM, BINS

YEARS = ("2022", "2023", "2024", "2025")
REGIME_COL = {"trend": "rg_trend", "vol": "rg_vol", "disaster": "rg_disaster"}
YEAR_CAP = 150_000          # cap rows/year so the permutation loop stays CPU-cheap


def _top_from_probe(k: int = 6) -> list[tuple]:
    """The top (feature, regime, target) interaction hits from MI2's output."""
    pj = OUT / "mi_probe.json"
    if not pj.exists():
        return [("dist_low_atr", "trend", "_fwd_r5"),
                ("onset_score", "trend", "_fwd_r5"),
                ("close_pct_prior", "trend", "_fwd_r5")]
    hits = json.loads(pj.read_text(encoding="utf-8"))["summary"]["hits"]
    out, seen = [], set()
    for h in hits:
        reg = h["regime"].replace("interact_", "")
        key = (h["feature"], reg, h["target"])
        if key not in seen:
            seen.add(key); out.append(key)
        if len(out) >= k:
            break
    return out


def per_year(df: pd.DataFrame, feature: str, regime: str, target: str,
             n_perm: int = N_PERM) -> dict:
    rcol = REGIME_COL[regime]
    res = {}
    for yr in YEARS:
        sub = df[df["trade_date"].str[:4] == yr]
        if len(sub) > YEAR_CAP:
            sub = sub.sample(YEAR_CAP, random_state=0)
        x = sub[feature].to_numpy(); y = sub[target].to_numpy()
        z = sub[rcol].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y, z = x[ok], y[ok], z[ok]
        if len(x) < 2000 or len(np.unique(z)) < 2:
            continue
        r = interaction_pvalue(x, y, z, n_perm=n_perm, bins=BINS)
        r["n"] = int(len(x))
        res[yr] = r
    return res


def run_real() -> dict:
    # build WITHOUT subsampling so each year keeps enough rows
    df = build(load_panel(), subsample=0)
    targets = sorted({t for _, _, t in _top_from_probe()})
    items = _top_from_probe(k=6)
    out_items = []
    for feat, reg, tgt in items:
        yr = per_year(df, feat, reg, tgt)
        n_pos_sig = sum(1 for v in yr.values()
                        if v["interaction"] > 0 and v["p_value"] < 0.05)
        n_years = len(yr)
        out_items.append({
            "feature": feat, "regime": reg, "target": tgt,
            "per_year": yr, "n_years": n_years,
            "n_pos_sig": n_pos_sig,
            "stable": n_years >= 2 and n_pos_sig >= max(2, int(np.ceil(2 * n_years / 3)))})
    n_stable = sum(1 for it in out_items if it["stable"])
    verdict = ("STABLE-CONDITIONAL-INFORMATION (build the motif model)"
               if n_stable >= 1 else
               "UNSTABLE (interaction not reproducible across years -- not a basis)")
    out = {"n_perm": N_PERM, "bins": BINS, "years": YEARS,
           "items": out_items, "n_stable": n_stable, "n_tested": len(out_items),
           "verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mi_stability.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    out = run_real()
    for it in out["items"]:
        ys = " ".join(f"{y}:{v['interaction']:+.4f}(p={v['p_value']:.3f})"
                      for y, v in it["per_year"].items())
        print(f"{it['feature']:<16}{it['regime']:<8}{it['target']:<12} "
              f"stable={it['stable']} pos_sig={it['n_pos_sig']}/{it['n_years']}  {ys}")
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
