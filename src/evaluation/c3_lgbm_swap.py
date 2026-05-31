"""C3 confirmatory — LGBM-swap test.

The dose-response experiment (c3_dose_response) showed that NO onset-rate or
expert-score composition of the WF pool reproduces the PoC's hybrid-beats-LGBM
reversal -> base rate is not the cause. The remaining hypothesis: the reversal
is an artifact of the PoC's WEAK LGBM baseline (e1_1_fh_h5_v2, +1.25% Top-10%),
which leaves headroom any LLM-boosted hybrid can fill.

Decisive test: hold the PoC anchors and their LLM scores FIXED, swap only the
LGBM base ranker, and check whether the hybrid's edge survives a stronger base.

  Model (a) e1_1_fh_h5_v2     -- the PoC's original single-window LGBM (weak).
  Model (b) wf_lgbm_split{2,3} -- per-split walk-forward LGBM (strong), mapped
                                  to each anchor by its trade_date window.

Restricted to the 583 PoC anchors inside the split2/3 test windows
[20250701, 20260101) where model (b) is legitimately out-of-sample.

If hybrid beats LGBM under (a) but not under (b), the reversal is a
weak-baseline artifact, not an LLM contribution -> C3 should be reframed as a
baseline-strength sensitivity, not base-rate compression.

Usage:
    .venv-xpu\\Scripts\\python.exe -m src.evaluation.c3_lgbm_swap
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.agent.eval_hybrid import add_lgbm_predictions          # model (a): e1_1
from src.agent.eval_wf_hybrid import add_lgbm, LGBM_MODELS, D1   # model (b): wf splits
from src.agent.hybrid_router import STRATEGIES

ROOT = Path(__file__).resolve().parents[2]
POC_PRED = ROOT / "results/poc_full/predictions.parquet"
OUT = ROOT / "results/c3_lgbm_swap"

WINDOWS = {2: ("20250701", "20251001"), 3: ("20251001", "20260101")}
BOOST = 0.30
N_BOOT = 1000
SEED = 42


def pctrank(a):
    return rankdata(a, method="average") / len(a)


def top10(sig, tgt):
    k = max(1, len(sig) // 10)
    idx = np.argpartition(-sig, k - 1)[:k]
    return float(tgt[idx].mean())


def rank_ic(sig, tgt):
    return float(np.corrcoef(pctrank(sig), pctrank(tgt))[0, 1])


def clustered_metrics(df: pd.DataFrame, lgbm_col: str) -> dict:
    """Date-clustered bootstrap of LGBM / hybrid / LLM Top-10% + RankIC and the
    hybrid-vs-LGBM delta, using lgbm_col as the base ranker."""
    work = df.copy()
    work["lgbm_pump_ratio"] = work[lgbm_col]
    hyb = STRATEGIES["E_lgbm_floor_llm_boost"](work, boost=BOOST).values
    sigs = {"LGBM": work[lgbm_col].values,
            "LLM_raw": work["raw_p_up"].values,
            "Hybrid_E": hyb}
    tgt = work["_fwd_r5"].values
    dates = work["trade_date"].values
    uniq = np.unique(dates)
    by_date = {d: np.where(dates == d)[0] for d in uniq}

    rng = np.random.default_rng(SEED)
    acc = {m: {"top10": [], "ic": []} for m in sigs}
    d_top10, d_ic = [], []
    for _ in range(N_BOOT):
        draw = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([by_date[d] for d in draw])
        t = tgt[rows]
        tk = {m: top10(sigs[m][rows], t) for m in sigs}
        ic = {m: rank_ic(sigs[m][rows], t) for m in sigs}
        for m in sigs:
            acc[m]["top10"].append(tk[m])
            acc[m]["ic"].append(ic[m])
        d_top10.append(tk["Hybrid_E"] - tk["LGBM"])
        d_ic.append(ic["Hybrid_E"] - ic["LGBM"])

    def ci(a):
        a = np.array(a)
        return {"mean": float(a.mean()), "lo": float(np.percentile(a, 2.5)),
                "hi": float(np.percentile(a, 97.5))}

    return {
        "point": {m: {"top10": top10(sigs[m], tgt), "rank_ic": rank_ic(sigs[m], tgt)} for m in sigs},
        "boot": {m: {"top10": ci(acc[m]["top10"]), "rank_ic": ci(acc[m]["ic"])} for m in sigs},
        "delta_hybrid_vs_lgbm": {"top10": ci(d_top10), "rank_ic": ci(d_ic)},
    }


def main():
    df_full = pd.read_parquet(POC_PRED)
    df_full["trade_date"] = df_full["trade_date"].astype(str)

    # --- Full 1000 PoC with the original weak LGBM: reproduce the +54% and
    #     give it a DATE-CLUSTERED CI (the paper used anchor-independent boot) ---
    df_full = add_lgbm_predictions(df_full)
    full_res = clustered_metrics(df_full, "lgbm_pump_ratio")

    df = df_full.copy()

    # restrict to split2/3 windows + tag split
    df["_wf_split"] = 0
    for sid, (lo, hi) in WINDOWS.items():
        df.loc[(df["trade_date"] >= lo) & (df["trade_date"] < hi), "_wf_split"] = sid
    df = df[df["_wf_split"] != 0].reset_index(drop=True)
    print(f"Restricted to {len(df)} PoC anchors in split2/3 windows "
          f"(split2={int((df['_wf_split']==2).sum())}, split3={int((df['_wf_split']==3).sum())})")

    # model (a): e1_1 (weak)
    df_a = add_lgbm_predictions(df)
    df["lgbm_a"] = df_a["lgbm_pump_ratio"].values

    # model (b): wf per-split (strong), date-mapped
    d1 = pd.read_parquet(D1)
    d1["trade_date"] = d1["trade_date"].astype(str)
    lgbm_b = np.full(len(df), np.nan)
    for sid in WINDOWS:
        m = (df["_wf_split"] == sid).values
        scored = add_lgbm(df[m], LGBM_MODELS[sid], d1)
        lgbm_b[m] = scored["lgbm_pump_ratio"].values
    df["lgbm_b"] = lgbm_b

    res = {"poc_full_1000_e1_1": full_res,
           "model_a_e1_1_weak_583": clustered_metrics(df, "lgbm_a"),
           "model_b_wf_strong_583": clustered_metrics(df, "lgbm_b")}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(
        {"n": int(len(df)), "n_boot": N_BOOT, "boost": BOOST, "results": res},
        indent=2), encoding="utf-8")

    # report
    for tag, r in res.items():
        print(f"\n=== {tag} ===")
        p, b = r["point"], r["boot"]
        print(f"  {'method':<10} {'Top10%':>8} {'RankIC':>9}   95% CI Top10%")
        for m in ("LGBM", "LLM_raw", "Hybrid_E"):
            print(f"  {m:<10} {p[m]['top10']*100:>+7.2f} {p[m]['rank_ic']:>+9.4f}   "
                  f"[{b[m]['top10']['lo']*100:+.2f}, {b[m]['top10']['hi']*100:+.2f}]")
        d = r["delta_hybrid_vs_lgbm"]["top10"]
        sig = "" if d["lo"] <= 0 <= d["hi"] else "  *significant*"
        print(f"  delta(Hybrid-LGBM) Top10% = {d['mean']*100:+.2f}pp  "
              f"95%CI [{d['lo']*100:+.2f}, {d['hi']*100:+.2f}]{sig}")

    print(f"\nWrote {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
