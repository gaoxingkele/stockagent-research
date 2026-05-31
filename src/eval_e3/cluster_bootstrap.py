"""E3 core — the C3 demonstration on a public LLM-finance benchmark.

Given per-anchor predictions for a TABULAR baseline and an LLM on a FLARE
stock-movement test split, this shows that the LLM-minus-baseline accuracy gap
typically reported as "the LLM helps" is:

  * apparently significant under an ANCHOR-INDEPENDENT bootstrap (the resampling
    LLM-finance papers commonly use, implicitly treating each stock-day as iid),
  * but NOT significant under a DATE-CLUSTERED bootstrap, because all stocks on
    the same trading day share the market move and are strongly correlated.

This is the field-level generalization of paper finding C3: at these sample
sizes, hybrid/LLM-over-baseline effects are within date-clustered noise, and
benchmarks that omit cluster-robust CIs manufacture spurious "LLM helps" claims.

The LLM predictions come from results/e3_<ds>/llm_test.parquet (column llm_p =
P(Rise)=P(gold... see below)). Until that exists, --demo-acc N synthesizes an
LLM with a controllable *true* accuracy so the machinery can be validated and
the naive-vs-clustered contrast illustrated; synthetic runs are clearly tagged.

Usage:
    .venv-xpu\\Scripts\\python.exe -m src.eval_e3.cluster_bootstrap --dataset acl
    .venv-xpu\\Scripts\\python.exe -m src.eval_e3.cluster_bootstrap --dataset acl --demo-acc 0.54
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
N_BOOT = 2000
SEED = 42


def synth_llm(gold: np.ndarray, target_acc: float, rng) -> np.ndarray:
    """Synthetic LLM prob with a controllable TRUE accuracy (for validation)."""
    correct = rng.random(len(gold)) < target_acc
    pred = np.where(correct, gold, 1 - gold)
    # map hard label to a soft prob with mild noise, kept on the correct side of 0.5
    return np.clip(pred * 0.5 + 0.25 + rng.normal(0, 0.08, len(gold)), 0.01, 0.99)


def acc_mcc(p, gold):
    pred = (np.asarray(p) >= 0.5).astype(int)
    return accuracy_score(gold, pred), matthews_corrcoef(gold, pred)


def boot_diff(gold, p_llm, p_base, dates, *, clustered: bool, rng):
    """Bootstrap distribution of (acc_llm - acc_base). clustered=True resamples
    whole trading days; clustered=False resamples individual anchors."""
    gold = np.asarray(gold); p_llm = np.asarray(p_llm); p_base = np.asarray(p_base)
    diffs = []
    if clustered:
        uniq = np.unique(dates)
        by_date = {d: np.where(dates == d)[0] for d in uniq}
        for _ in range(N_BOOT):
            draw = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([by_date[d] for d in draw])
            a_l, _ = acc_mcc(p_llm[idx], gold[idx])
            a_b, _ = acc_mcc(p_base[idx], gold[idx])
            diffs.append(a_l - a_b)
    else:
        n = len(gold)
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            a_l, _ = acc_mcc(p_llm[idx], gold[idx])
            a_b, _ = acc_mcc(p_base[idx], gold[idx])
            diffs.append(a_l - a_b)
    d = np.array(diffs)
    return {"mean": float(d.mean()),
            "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
            "p_two_sided_ge0": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="acl")
    ap.add_argument("--demo-acc", type=float, default=None,
                    help="synthesize an LLM with this true accuracy (validation only)")
    args = ap.parse_args()

    out_dir = RESULTS / f"e3_{args.dataset}"
    base = pd.read_parquet(out_dir / "baseline_test.parquet")
    rng = np.random.default_rng(SEED)

    synthetic = False
    llm_path = out_dir / "llm_test.parquet"
    if args.demo_acc is not None:
        df = base.copy()
        df["llm_p"] = synth_llm(df["gold"].values, args.demo_acc, rng)
        synthetic = True
    elif llm_path.exists():
        llm = pd.read_parquet(llm_path)[["id", "llm_p"]]
        df = base.merge(llm, on="id", how="inner")
    else:
        raise SystemExit(f"No {llm_path}; run the LLM step or pass --demo-acc for validation.")

    gold = df["gold"].values
    dates = df["date"].values
    a_b, m_b = acc_mcc(df["baseline_p"], gold)
    a_l, m_l = acc_mcc(df["llm_p"], gold)

    naive = boot_diff(gold, df["llm_p"], df["baseline_p"], dates, clustered=False, rng=rng)
    clust = boot_diff(gold, df["llm_p"], df["baseline_p"], dates, clustered=True, rng=rng)

    tag = " [SYNTHETIC LLM]" if synthetic else ""
    print(f"\n=== E3 / {args.dataset}{tag} ===")
    print(f"n={len(df)}  unique dates={df['date'].nunique()}  (~{len(df)//max(1,df['date'].nunique())}/day)")
    print(f"baseline accuracy={a_b:.4f}  MCC={m_b:+.4f}")
    print(f"LLM      accuracy={a_l:.4f}  MCC={m_l:+.4f}")
    print(f"LLM - baseline accuracy gap = {(a_l-a_b)*100:+.2f}pp")
    def line(name, r):
        sig = "EXCLUDES 0 (looks significant)" if (r["lo"] > 0 or r["hi"] < 0) else "spans 0 (NOT significant)"
        print(f"  {name:<26} mean {r['mean']*100:+.2f}pp  95% CI [{r['lo']*100:+.2f}, {r['hi']*100:+.2f}]pp  p={r['p_two_sided_ge0']:.3f}  -> {sig}")
    print("Bootstrap CI on the accuracy gap:")
    line("anchor-independent", naive)
    line("date-clustered", clust)
    width_ratio = (clust["hi"] - clust["lo"]) / max(1e-9, (naive["hi"] - naive["lo"]))
    print(f"  clustered CI is {width_ratio:.2f}x wider than the naive CI")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "c3_bootstrap.json").write_text(json.dumps({
        "dataset": args.dataset, "synthetic_llm": synthetic, "n": int(len(df)),
        "n_dates": int(df["date"].nunique()), "n_boot": N_BOOT,
        "baseline": {"acc": a_b, "mcc": m_b}, "llm": {"acc": a_l, "mcc": m_l},
        "gap_acc_pp": (a_l - a_b) * 100,
        "naive_bootstrap": naive, "clustered_bootstrap": clust,
        "ci_width_ratio_clustered_over_naive": width_ratio,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'c3_bootstrap.json'}")


if __name__ == "__main__":
    main()
