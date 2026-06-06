"""TRD1 -- REAL ($0): is the stable conditional information DIRECTIONAL or sign-blind?

Applies the SGN1 decomposer to the MI3 winners (the feature x regime hits whose
interaction information was cross-period stable). Mutual information said the trend
regime ADDS information; this asks the question MI couldn't: is that information in
the conditional MEAN (directional, tradable long) or only in the conditional
VARIANCE (sign-blind risk)? And is the conditional-mean response MONOTONE?

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_tradability
"""
from __future__ import annotations

import json

import numpy as np

from src.onset.run_mi_probe import load_panel, build, OUT
from src.onset.tradability import directionality, monotonicity, variance_vs_mean

REGIME_COL = {"trend": "rg_trend", "vol": "rg_vol", "disaster": "rg_disaster"}


def _hits(k: int = 6) -> list[tuple]:
    sj = OUT / "mi_stability.json"
    if sj.exists():
        items = json.loads(sj.read_text(encoding="utf-8"))["items"]
        return [(it["feature"], it["regime"], it["target"])
                for it in items if it.get("stable")][:k] or \
               [(it["feature"], it["regime"], it["target"]) for it in items][:k]
    return [("close_pct_prior", "trend", "_fwd_r5"),
            ("dist_low_atr", "trend", "_fwd_r5")]


def diagnose(df, feature: str, regime: str, target: str) -> dict:
    x = df[feature].to_numpy(); y = df[target].to_numpy()
    z = df[REGIME_COL[regime]].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, z = x[ok], y[ok], z[ok]
    d = directionality(x, y, z)
    m = monotonicity(x, y, z)
    v = variance_vs_mean(x, y, z)
    # A gated long-only strategy trades WITHIN one regime state, so per-state
    # magnitude is what matters -- the overall slope can cancel when the sign flips
    # across states. Take the strongest single state.
    best_slope = max((abs(s["slope"]) for s in d["per_state"].values()), default=0.0)
    best_mono = max((abs(s["mono_coef"]) for s in m["per_state"].values()), default=0.0)
    tradable = (best_slope > 0.03 and best_mono > 0.6 and
                v["directional_fraction"] > 0.4)
    return {"feature": feature, "regime": regime, "target": target,
            "directionality": d, "monotonicity": m, "variance_vs_mean": v,
            "best_state_slope": float(best_slope), "best_state_mono": float(best_mono),
            "looks_directional_tradable": bool(tradable)}


def run_real() -> dict:
    df = build(load_panel())          # candle feats + regimes, subsampled
    items = _hits()
    diags = [diagnose(df, f, r, t) for f, r, t in items]
    n_dir = sum(1 for d in diags if d["looks_directional_tradable"])
    verdict = ("SOME-DIRECTIONAL" if n_dir else
               "ALL-SIGN-BLIND-OR-NONMONOTONE (information real but not directly long-tradable)")
    out = {"n_items": len(diags), "n_directional": n_dir,
           "verdict": verdict, "items": diags}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tradability.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    out = run_real()
    for d in out["items"]:
        print(f"{d['feature']:<16}{d['regime']:<8} best_slope={d['best_state_slope']:.3f} "
              f"best_mono={d['best_state_mono']:.3f} "
              f"dir_frac={d['variance_vs_mean']['directional_fraction']:.2f} "
              f"tradable={d['looks_directional_tradable']}")
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
