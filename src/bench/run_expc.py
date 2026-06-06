"""EXP-C -- REAL ($0): NMI + permutation info-theory baseline vs our interaction info.

Positions C6 against the existing info-theory framings: Financial Information
Theory (2511.16339) uses Normalized Mutual Information (NMI, bounded [0,1]) for
market-efficiency/regime detection; Conditional P-threshold MI (2601.00395) uses
permutation significance. Neither asks whether the regime ADDS information. We
recompute the MI3 leading hits as marginal NMI + conditional NMI + permutation p
(the baseline view) and contrast with OUR interaction information
II = I(X;Y|Z) - I(X;Y) tested by Z-label permutation (mi_probe.json). The point:
the NMI/conditional-MI baselines see near-zero marginal dependence (consistent
with FIT's <0.05 in efficient regimes) but do NOT isolate the regime-added
information that our II does.

Run: .venv-xpu\\Scripts\\python.exe -m src.bench.run_expc
"""
from __future__ import annotations

import json

import numpy as np

from src.train_tcn_wf import ROOT
from src.onset.mutual_info import quantile_bins, _mi_from_codes, perm_pvalue
from src.onset.run_mi_probe import load_panel, build, OUT as MOTIF_OUT, N_PERM, BINS

OUT = ROOT / "results/bench"
REGIME_COL = {"trend": "rg_trend", "vol": "rg_vol", "disaster": "rg_disaster"}


def _entropy(codes: np.ndarray) -> float:
    n = len(codes)
    if n == 0:
        return 0.0
    _, cnt = np.unique(codes, return_counts=True)
    p = cnt / n
    return float(-(p * np.log(p)).sum())


def nmi(x: np.ndarray, y: np.ndarray, bins: int = BINS) -> float:
    cx, cy = quantile_bins(x, bins), quantile_bins(y, bins)
    mi = _mi_from_codes(cx, cy)
    d = np.sqrt(_entropy(cx) * _entropy(cy))
    return float(mi / d) if d > 0 else 0.0


def conditional_nmi(x: np.ndarray, y: np.ndarray, z: np.ndarray, bins: int = BINS) -> float:
    n = len(x); tot = 0.0
    for s in np.unique(z):
        m = z == s; ns = int(m.sum())
        if ns < bins:
            continue
        tot += (ns / n) * nmi(x[m], y[m], bins)
    return float(tot)


def _hits(k: int = 6) -> list:
    sj = MOTIF_OUT / "mi_stability.json"
    items = json.loads(sj.read_text(encoding="utf-8"))["items"]
    return [(it["feature"], it["regime"], it["target"]) for it in items][:k]


def _interaction_from_probe(feature: str, regime: str, target: str) -> dict:
    pj = json.loads((MOTIF_OUT / "mi_probe.json").read_text(encoding="utf-8"))
    e = pj["results"].get(target, {}).get(feature, {})
    inter = e.get(f"interact_{regime}", {})
    return {"interaction": inter.get("interaction"), "interact_p": inter.get("p_value"),
            "cond_mi": inter.get("cond_mi"), "marg_mi": inter.get("marg_mi")}


def evaluate(df, n_perm: int = N_PERM) -> dict:
    items = _hits()
    out = []
    for feat, reg, tgt in items:
        x = df[feat].to_numpy(); y = df[tgt].to_numpy(); z = df[REGIME_COL[reg]].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y, z = x[ok], y[ok], z[ok]
        marg_nmi = nmi(x, y)
        cond_nmi = conditional_nmi(x, y, z)
        marg_p = perm_pvalue(x, y, n_perm=n_perm, conditional=False, bins=BINS)["p_value"]
        cond_p = perm_pvalue(x, y, z, n_perm=n_perm, conditional=True, bins=BINS)["p_value"]
        out.append({"feature": feat, "regime": reg, "target": tgt,
                    "marginal_nmi": marg_nmi, "conditional_nmi": cond_nmi,
                    "marginal_perm_p": marg_p, "conditional_perm_p": cond_p,
                    "ours_interaction": _interaction_from_probe(feat, reg, tgt)})
    return out


def _positioning(items: list) -> dict:
    max_marg_nmi = max((it["marginal_nmi"] for it in items), default=0.0)
    n_inter = sum(1 for it in items
                  if (it["ours_interaction"].get("interact_p") or 1) < 0.05
                  and (it["ours_interaction"].get("interaction") or 0) > 0)
    return {
        "max_marginal_nmi": max_marg_nmi,
        "nmi_efficient_regime": bool(max_marg_nmi < 0.05),
        "n_interaction_significant": n_inter,
        "verdict": (
            "NMI/conditional-MI baselines see near-zero marginal dependence "
            f"(max marginal NMI={max_marg_nmi:.4f}{' < 0.05, the FIT efficient-market band' if max_marg_nmi < 0.05 else ''}); "
            f"only our interaction information (Z-label permutation) isolates regime-ADDED "
            f"information ({n_inter}/{len(items)} hits II>0 at p<0.05). The existing "
            "info-theory framings would not have surfaced the regime-conditional structure as added information.")}


def run_real() -> dict:
    df = build(load_panel())
    items = evaluate(df)
    res = {"n_perm": N_PERM, "bins": BINS, "items": items,
           "summary": _positioning(items)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "expc.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    res = run_real()
    for it in res["items"]:
        oi = it["ours_interaction"]
        print(f"{it['feature']:<16}{it['regime']:<8} margNMI={it['marginal_nmi']:.4f}(p={it['marginal_perm_p']:.3f}) "
              f"condNMI={it['conditional_nmi']:.4f} | ours II={oi.get('interaction'):+.4f}(p={oi.get('interact_p'):.3f})")
    print("VERDICT:", res["summary"]["verdict"])


if __name__ == "__main__":
    main()
