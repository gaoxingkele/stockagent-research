"""C3 strengthening — E0 + E1 + E4, decomposing the stratified-vs-random gap.

The paper's C3 compares ONE stratified PoC (n=1000, 25% onset, oversampled by
expert onset_score) against ONE walk-forward random sample (n=6000, 8% onset)
and attributes the LLM/hybrid advantage reversal to base-rate compression.
That comparison confounds THREE things: marginal onset rate, the
stratification *structure* (oversampling high expert-score anchors -- a
feature the LLM consumes), and sample size.

This script holds the 6,000 already-scored WF anchors fixed (E0: no new n /
anchor-set / scoring confound) and runs TWO controlled sweeps to separate the
two candidate mechanisms:

  Rate sweep      : vary the marginal onset rate (is_bullish_onset fraction),
                    subsampling uniformly within each class.
  Structure sweep : vary the fraction of expert-positive (onset_score >= 3)
                    anchors -- i.e. reconstruct the PoC stratification axis,
                    which conditions on an LLM input feature.

For each grid point we report Delta(method - LGBM) Top-10% return + RankIC for
pure LLM (raw/expert) and the E_lgbm_floor_llm_boost hybrid (the +54% claim),
with date-CLUSTERED bootstrap CIs (E4) -- anchors on the same trade_date are
correlated, which the anchor-independent bootstrap in eval_wf_hybrid ignores.

Reuses add_lgbm / SPLITS / LGBM_MODELS / D1 from eval_wf_hybrid so the LGBM
signal and Top-K definition match Table 1/2 exactly.

Usage:
    .venv-xpu\\Scripts\\python.exe -m src.evaluation.c3_dose_response
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.agent.eval_wf_hybrid import D1, SPLITS, LGBM_MODELS, add_lgbm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/c3_dose_response"
FIG_DIR = ROOT / "paper/sections/figures"

DATE = "trade_date"
TARGET = "_fwd_r5"
ONSET = "_exp_is_bullish_onset"
SCORE = "_exp_onset_score"

GRID = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.33, 0.40, 0.50]
DEPLOY_RATE = 0.08
POC_RATE = 0.25
BOOST = 0.30
LLM_TOPK_Q = 0.75
N_BOOT = 400
SEED = 42
TOPK_FRAC = 0.10


def load_pool() -> pd.DataFrame:
    d1 = pd.read_parquet(D1)
    d1[DATE] = d1[DATE].astype(str)
    parts = []
    for sid, path in SPLITS.items():
        df = pd.read_parquet(path)
        df = add_lgbm(df, LGBM_MODELS[sid], d1)
        parts.append(df)
    pool = pd.concat(parts, ignore_index=True)
    cols = [DATE, ONSET, SCORE, TARGET,
            "lgbm_pump_ratio", "raw_pump_ratio", "expert_pump_ratio", "raw_p_up"]
    pool = pool[cols].dropna(
        subset=[TARGET, "lgbm_pump_ratio", "raw_pump_ratio", "expert_pump_ratio", "raw_p_up"]
    ).reset_index(drop=True)
    pool[ONSET] = pool[ONSET].astype(bool)
    return pool


def pctrank(a: np.ndarray) -> np.ndarray:
    return rankdata(a, method="average") / len(a)


def top_k_return(sig: np.ndarray, tgt: np.ndarray, frac: float) -> float:
    k = max(1, int(len(sig) * frac))
    idx = np.argpartition(-sig, k - 1)[:k]
    return float(tgt[idx].mean())


def rank_ic(sig: np.ndarray, tgt: np.ndarray) -> float:
    return float(np.corrcoef(pctrank(sig), pctrank(tgt))[0, 1])


def signals_for(sel, lgbm, raw_ratio, exp_ratio, raw_pup):
    """Compute all method scores on a selected subsample (ranks recomputed)."""
    e_boost = pctrank(lgbm[sel]) + (pctrank(raw_pup[sel]) >= LLM_TOPK_Q).astype(float) * BOOST
    return {
        "LGBM": lgbm[sel],
        "LLM_raw": raw_ratio[sel],
        "LLM_expert": exp_ratio[sel],
        "Hybrid_E": e_boost,
    }


METHODS = ["LGBM", "LLM_raw", "LLM_expert", "Hybrid_E"]
VS_LGBM = ["LLM_raw", "LLM_expert", "Hybrid_E"]


def subsample_to_frac(target_idx, other_idx, f, rng):
    nT, nO = len(target_idx), len(other_idx)
    if nT == 0 or nO == 0:
        return np.concatenate([target_idx, other_idx])
    other_needed = int(round(nT * (1 - f) / f))
    if other_needed <= nO:
        other_idx = rng.choice(other_idx, other_needed, replace=False)
    else:
        t_needed = max(1, int(round(nO * f / (1 - f))))
        target_idx = rng.choice(target_idx, t_needed, replace=False)
    return np.concatenate([target_idx, other_idx])


def sweep(pool: pd.DataFrame, strat_mask: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    dates = pool[DATE].values
    tgt = pool[TARGET].values
    lgbm = pool["lgbm_pump_ratio"].values
    raw_ratio = pool["raw_pump_ratio"].values
    exp_ratio = pool["expert_pump_ratio"].values
    raw_pup = pool["raw_p_up"].values

    uniq = pool[DATE].unique()
    by_date = {d: np.where(dates == d)[0] for d in uniq}

    out = {}
    for f in GRID:
        d_top10 = {m: [] for m in VS_LGBM}
        d_ric = {m: [] for m in VS_LGBM}
        abs_top10 = {m: [] for m in METHODS}
        ns, fracs = [], []
        for _ in range(N_BOOT):
            draw = rng.choice(uniq, len(uniq), replace=True)
            rows = np.concatenate([by_date[d] for d in draw])
            mask = strat_mask[rows]
            sel = subsample_to_frac(rows[mask], rows[~mask], f, rng)
            t = tgt[sel]
            ns.append(len(sel))
            fracs.append(float(strat_mask[sel].mean()))
            sigs = signals_for(sel, lgbm, raw_ratio, exp_ratio, raw_pup)
            tk = {m: top_k_return(sigs[m], t, TOPK_FRAC) for m in METHODS}
            ic = {m: rank_ic(sigs[m], t) for m in METHODS}
            for m in METHODS:
                abs_top10[m].append(tk[m])
            for m in VS_LGBM:
                d_top10[m].append(tk[m] - tk["LGBM"])
                d_ric[m].append(ic[m] - ic["LGBM"])

        def ci(a):
            a = np.array(a)
            return {"mean": float(a.mean()),
                    "lo": float(np.percentile(a, 2.5)),
                    "hi": float(np.percentile(a, 97.5))}

        out[f] = {
            "frac_actual": float(np.mean(fracs)),
            "n_mean": float(np.mean(ns)),
            "abs_top10": {m: ci(abs_top10[m]) for m in METHODS},
            "delta_top10": {m: ci(d_top10[m]) for m in VS_LGBM},
            "delta_rank_ic": {m: ci(d_ric[m]) for m in VS_LGBM},
        }
    return out


def crossover(sweep_res, method, metric="delta_top10"):
    xs = sorted(sweep_res)
    ys = [sweep_res[r][metric][method]["mean"] for r in xs]
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 <= 0 <= y1) or (y0 >= 0 >= y1):
            if y1 == y0:
                return xs[i]
            return xs[i] + (xs[i + 1] - xs[i]) * (0 - y0) / (y1 - y0)
    return None


def plot(rate_res, struct_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"LLM_raw": "#1f77b4", "LLM_expert": "#d62728", "Hybrid_E": "#9467bd"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (res, title, vline, vlabel) in zip(
        axes,
        [(rate_res, "A. Rate sweep — vary onset (is_bullish) rate", DEPLOY_RATE, "deploy 8%"),
         (struct_res, "B. Structure sweep — vary expert-score≥3 fraction", POC_RATE, "PoC 25%")],
    ):
        xs = np.array(sorted(res)) * 100
        for m in VS_LGBM:
            mean = np.array([res[r]["delta_top10"][m]["mean"] for r in sorted(res)]) * 100
            lo = np.array([res[r]["delta_top10"][m]["lo"] for r in sorted(res)]) * 100
            hi = np.array([res[r]["delta_top10"][m]["hi"] for r in sorted(res)]) * 100
            ax.plot(xs, mean, "o-", color=colors[m], label=m, ms=4)
            ax.fill_between(xs, lo, hi, color=colors[m], alpha=0.12)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(vline * 100, color="gray", ls="--", lw=1, label=vlabel)
        ax.set_xlabel("evaluation fraction (%)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\Delta$ Top-10% return vs LGBM (pp)")
    fig.suptitle("C3 decomposition: only expert-score stratification reproduces the LLM/hybrid advantage",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "c3_dose_response.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote figure -> {out}")


def _print_sweep(name, res):
    print(f"\n=== {name} ===")
    print(f"{'frac%':>6} {'n':>6} {'LGBM':>7} {'raw':>7} {'expert':>7} {'hybrid':>7}  "
          f"{'Dhyb':>7} {'Draw':>7}  (Top-10% ret, pp)")
    for r in sorted(res):
        d = res[r]
        a = d["abs_top10"]
        print(f"{d['frac_actual']*100:>5.1f}% {d['n_mean']:>6.0f} "
              f"{a['LGBM']['mean']*100:>+6.2f} {a['LLM_raw']['mean']*100:>+6.2f} "
              f"{a['LLM_expert']['mean']*100:>+6.2f} {a['Hybrid_E']['mean']*100:>+6.2f}  "
              f"{d['delta_top10']['Hybrid_E']['mean']*100:>+6.2f} "
              f"{d['delta_top10']['LLM_raw']['mean']*100:>+6.2f}")


def main():
    pool = load_pool()
    onset_mask = pool[ONSET].values
    score_mask = (pool[SCORE].values >= 3)
    print(f"Pool: {len(pool)} anchors, {len(pool[DATE].unique())} dates | "
          f"onset rate {onset_mask.mean():.1%} ({onset_mask.sum()} pos) | "
          f"expert-score>=3 {score_mask.mean():.1%} ({score_mask.sum()})")

    rate_res = sweep(pool, onset_mask)
    struct_res = sweep(pool, score_mask)

    _print_sweep("RATE sweep (vary is_bullish_onset rate)", rate_res)
    _print_sweep("STRUCTURE sweep (vary onset_score>=3 fraction)", struct_res)

    cross = {
        "rate": {m: crossover(rate_res, m) for m in VS_LGBM},
        "structure": {m: crossover(struct_res, m) for m in VS_LGBM},
    }
    print("\nCrossover (Δ Top-10% = 0):")
    for sweep_name, d in cross.items():
        for m, x in d.items():
            print(f"  {sweep_name:>9} / {m:<11}: "
                  + (f"{x*100:.1f}%" if x is not None else "none in grid"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps({
        "n_pool": int(len(pool)), "n_boot": N_BOOT, "topk_frac": TOPK_FRAC,
        "boost": BOOST, "crossover": cross,
        "rate_sweep": rate_res, "structure_sweep": struct_res,
    }, indent=2), encoding="utf-8")
    plot(rate_res, struct_res)
    print(f"\nWrote {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
