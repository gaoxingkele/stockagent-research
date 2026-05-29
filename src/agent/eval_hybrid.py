"""Evaluate hybrid routing strategies vs baselines on the 1000-sample PoC.

Outputs:
  - results/hybrid_eval/comparison.parquet  (per-anchor predictions + signals)
  - results/hybrid_eval/metrics.json         (all metric variants)
  - results/hybrid_eval/summary.md           (Markdown report)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

from src.agent.hybrid_router import STRATEGIES, _rankify

ROOT = Path(__file__).resolve().parents[2]
POC_PRED = ROOT / "results/poc_full/predictions.parquet"
D1 = ROOT / "data/processed/ashares_tushare_v1.parquet"
LGBM_MODEL = ROOT / "results/e1_1_fh_h5_v2/model.txt"
OUT = ROOT / "results/hybrid_eval"
OUT.mkdir(parents=True, exist_ok=True)


NON_FEAT = {
    "ts_code", "trade_date", "industry",
    "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount",
    "r5", "r10", "r20", "r30", "r40",
    "dd5", "dd10", "dd20", "dd30", "dd40",
}


def add_lgbm_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Score the 1000 anchors with the trained LGBM, attach prob columns."""
    booster = lgb.Booster(model_file=str(LGBM_MODEL))
    d1 = pd.read_parquet(D1)
    d1["trade_date"] = d1["trade_date"].astype(str)
    key = df[["ts_code", "trade_date"]]
    d1_anchored = key.merge(d1, on=["ts_code", "trade_date"], how="left")
    feat_cols = [
        c for c in d1_anchored.columns
        if c not in NON_FEAT
        and pd.api.types.is_numeric_dtype(d1_anchored[c])
        and d1_anchored[c].notna().mean() >= 0.05
    ]
    X = d1_anchored[feat_cols]
    prob = booster.predict(X, num_iteration=booster.best_iteration)
    df = df.copy()
    df["lgbm_p_down"] = prob[:, 0]
    df["lgbm_p_neutral"] = prob[:, 1]
    df["lgbm_p_up"] = prob[:, 2]
    df["lgbm_pump_ratio"] = prob[:, 2] / (prob[:, 0] + 0.01)
    df["lgbm_max_prob"] = prob.max(axis=1)
    return df


def evaluate(signal: pd.Series, target: pd.Series) -> dict:
    sub = pd.DataFrame({"sig": signal, "tgt": target}).dropna()
    if len(sub) < 10:
        return {"n": len(sub)}
    rho, p = spearmanr(sub["sig"], sub["tgt"])
    sorted_sub = sub.sort_values("sig", ascending=False)
    top10 = sorted_sub.head(max(1, len(sub) // 10))
    top20 = sorted_sub.head(max(1, len(sub) // 5))
    return {
        "n": int(len(sub)),
        "rank_ic": float(rho),
        "rank_ic_p": float(p),
        "top10_mean": float(top10["tgt"].mean()),
        "top10_winrate": float((top10["tgt"] > 0).mean()),
        "top20_mean": float(top20["tgt"].mean()),
        "top20_winrate": float((top20["tgt"] > 0).mean()),
    }


def main():
    df = pd.read_parquet(POC_PRED)
    print(f"Loaded {len(df)} PoC predictions")
    df = add_lgbm_predictions(df)
    print(f"Attached LGBM predictions (165 features)")

    # Baselines
    runs = {}
    runs["BL_LGBM"] = df["lgbm_pump_ratio"]
    runs["BL_LLM_raw_p_up"] = df["raw_p_up"]
    runs["BL_LLM_expert_p_up"] = df["expert_p_up"]
    runs["BL_LLM_raw_pump_ratio"] = df["raw_pump_ratio"]
    runs["BL_LLM_expert_pump_ratio"] = df["expert_pump_ratio"]
    runs["BL_expert_rule"] = df["_exp_onset_score"].astype(float)

    # Hybrid strategies (sweeping key hyperparams)
    runs["H_A_conf_0.3"] = STRATEGIES["A_confidence"](df, conf_threshold=0.3)
    runs["H_A_conf_0.5"] = STRATEGIES["A_confidence"](df, conf_threshold=0.5)
    runs["H_A_conf_0.7"] = STRATEGIES["A_confidence"](df, conf_threshold=0.7)
    runs["H_A_conf_maxprob_0.5"] = STRATEGIES["A_confidence"](
        df, lgbm_conf_col="lgbm_max_prob", conf_threshold=0.5
    )

    runs["H_B_stratum"] = STRATEGIES["B_stratum"](df)
    runs["H_B_stratum_thresh2"] = STRATEGIES["B_stratum"](df, onset_threshold=2)

    runs["H_C_soft_w0.5"] = STRATEGIES["C_soft_ensemble"](df, lgbm_weight=0.5)
    runs["H_C_soft_w0.6"] = STRATEGIES["C_soft_ensemble"](df, lgbm_weight=0.6)
    runs["H_C_soft_w0.7"] = STRATEGIES["C_soft_ensemble"](df, lgbm_weight=0.7)
    runs["H_C_soft_w0.8"] = STRATEGIES["C_soft_ensemble"](df, lgbm_weight=0.8)

    runs["H_D_avoid_expert"] = STRATEGIES["D_avoid_expert_onset"](df)

    runs["H_E_lgbm_floor_boost0.15"] = STRATEGIES["E_lgbm_floor_llm_boost"](
        df, boost=0.15
    )
    runs["H_E_lgbm_floor_boost0.30"] = STRATEGIES["E_lgbm_floor_llm_boost"](
        df, boost=0.30
    )

    # Evaluate all
    target = df["_fwd_r5"]
    all_metrics = {}
    print(f"\n{'Method':<35} {'n':>5} {'RankIC':>9} {'Top10%':>8} {'WR10':>6} {'Top20%':>8} {'WR20':>6}")
    print("-" * 90)
    for name, sig in runs.items():
        m = evaluate(sig, target)
        all_metrics[name] = m
        print(f"{name:<35} {m['n']:>5} {m['rank_ic']:>+9.4f} "
              f"{m['top10_mean']*100:>+7.2f}% {m['top10_winrate']:>6.1%} "
              f"{m['top20_mean']*100:>+7.2f}% {m['top20_winrate']:>6.1%}")

    # By-stratum analysis for top performers
    print("\n=== By stratum (selected methods) ===")
    selected = ["BL_LGBM", "BL_LLM_raw_p_up", "BL_LLM_expert_p_up",
                 "H_A_conf_0.5", "H_B_stratum", "H_C_soft_w0.6",
                 "H_D_avoid_expert", "H_E_lgbm_floor_boost0.30"]
    strata_records = []
    for s in ["high", "edge", "low"]:
        mask = df["stratum"] == s
        sub_df = df[mask].copy()
        sub_target = target[mask]
        print(f"\nStratum '{s}' (n={int(mask.sum())}, mean fwd_r5={sub_target.mean()*100:+.2f}%):")
        for name in selected:
            if name in runs:
                m = evaluate(runs[name][mask], sub_target)
                strata_records.append({"method": name, "stratum": s, **m})
                print(f"  {name:<35} Top10={m['top10_mean']*100:>+6.2f}%  "
                      f"WR={m['top10_winrate']:>5.1%}  RankIC={m['rank_ic']:>+.3f}")

    # Save
    pd.DataFrame(runs).to_parquet(OUT / "all_signals.parquet")
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"main": all_metrics,
                    "by_stratum": strata_records}, f, indent=2)

    # Summary md
    lines = ["# Hybrid Strategy Evaluation\n",
             f"- n = {len(df)} (Sonnet 4.6 PoC anchors)",
             f"- baselines + 15 hybrid variants\n",
             "## Overall comparison\n",
             "| Method | n | RankIC | Top10% | WR10 | Top20% | WR20 |",
             "|---|---|---|---|---|---|---|"]
    for name, m in all_metrics.items():
        lines.append(
            f"| `{name}` | {m.get('n','-')} | "
            f"{m.get('rank_ic', float('nan')):+.4f} | "
            f"{m.get('top10_mean', float('nan'))*100:+.2f}% | "
            f"{m.get('top10_winrate', float('nan')):.1%} | "
            f"{m.get('top20_mean', float('nan'))*100:+.2f}% | "
            f"{m.get('top20_winrate', float('nan')):.1%} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
