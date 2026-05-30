"""W1.5 — Walk-forward hybrid evaluation across 3 splits with bootstrap CI.

For each split:
  - Load 2000 anchor LLM predictions (raw + expert) from poc_wf_split{1,2,3}
  - Load LGBM model trained for that split, predict probabilities
  - Compute baseline + hybrid signals
  - Evaluate (RankIC, Top10%, Top20%) on the same anchors
  - Bootstrap 1000-resample 95% CI on Top10%/Top20% return

Aggregate across splits:
  - Mean ± std of each metric
  - DM-test-style: paired comparison H_E vs LGBM per split
  - Bootstrap CI for the aggregate
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
OUT = ROOT / "results/wf_hybrid_eval"
OUT.mkdir(parents=True, exist_ok=True)

NON_FEAT = {
    "ts_code", "trade_date", "industry",
    "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount",
    "r5", "r10", "r20", "r30", "r40",
    "dd5", "dd10", "dd20", "dd30", "dd40",
}

D1 = ROOT / "data/processed/ashares_tushare_v1.parquet"

SPLITS = {
    1: ROOT / "results/poc_wf_split1/predictions.parquet",
    2: ROOT / "results/poc_wf_split2/predictions.parquet",
    3: ROOT / "results/poc_wf_split3/predictions.parquet",
}
LGBM_MODELS = {
    1: ROOT / "results/wf_lgbm_split1/model.txt",
    2: ROOT / "results/wf_lgbm_split2/model.txt",
    3: ROOT / "results/wf_lgbm_split3/model.txt",
}


def add_lgbm(df: pd.DataFrame, model_path: Path, d1: pd.DataFrame) -> pd.DataFrame:
    booster = lgb.Booster(model_file=str(model_path))
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
        return {"n": int(len(sub))}
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


def bootstrap_ci(signal: pd.Series, target: pd.Series, n_resample: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    sub = pd.DataFrame({"sig": signal, "tgt": target}).dropna()
    if len(sub) < 100:
        return {}
    top10_means = []
    top20_means = []
    rank_ics = []
    n = len(sub)
    for _ in range(n_resample):
        idx = rng.integers(0, n, size=n)
        sample = sub.iloc[idx]
        sorted_s = sample.sort_values("sig", ascending=False)
        top10_means.append(sorted_s.head(max(1, n // 10))["tgt"].mean())
        top20_means.append(sorted_s.head(max(1, n // 5))["tgt"].mean())
        rho, _ = spearmanr(sample["sig"], sample["tgt"])
        rank_ics.append(rho if not np.isnan(rho) else 0.0)
    return {
        "top10_ci95": (float(np.percentile(top10_means, 2.5)),
                       float(np.percentile(top10_means, 97.5))),
        "top10_mean_bootstrap": float(np.mean(top10_means)),
        "top20_ci95": (float(np.percentile(top20_means, 2.5)),
                       float(np.percentile(top20_means, 97.5))),
        "rank_ic_ci95": (float(np.percentile(rank_ics, 2.5)),
                         float(np.percentile(rank_ics, 97.5))),
    }


def main():
    print("Loading D1...")
    d1 = pd.read_parquet(D1)
    d1["trade_date"] = d1["trade_date"].astype(str)
    print(f"D1: {len(d1):,} rows")

    per_split = {}
    pooled_dfs = []

    for split_id, pred_path in SPLITS.items():
        print(f"\n=== Split {split_id} ===")
        df = pd.read_parquet(pred_path)
        df = add_lgbm(df, LGBM_MODELS[split_id], d1)
        print(f"  loaded {len(df)} predictions + LGBM scores")

        # Baselines + hybrid signals
        runs = {
            "BL_LGBM": df["lgbm_pump_ratio"],
            "BL_LLM_raw": df["raw_p_up"],
            "BL_LLM_expert": df["expert_p_up"],
            "BL_LLM_raw_ratio": df["raw_pump_ratio"],
            "BL_LLM_expert_ratio": df["expert_pump_ratio"],
            "BL_expert_rule": df["_exp_onset_score"].astype(float),
            "H_A_conf_0.5": STRATEGIES["A_confidence"](df, conf_threshold=0.5),
            "H_A_conf_maxprob_0.5": STRATEGIES["A_confidence"](
                df, lgbm_conf_col="lgbm_max_prob", conf_threshold=0.5),
            "H_B_stratum": STRATEGIES["B_stratum"](df),
            "H_C_soft_w0.6": STRATEGIES["C_soft_ensemble"](df, lgbm_weight=0.6),
            "H_D_avoid_expert": STRATEGIES["D_avoid_expert_onset"](df),
            "H_E_boost_0.15": STRATEGIES["E_lgbm_floor_llm_boost"](df, boost=0.15),
            "H_E_boost_0.30": STRATEGIES["E_lgbm_floor_llm_boost"](df, boost=0.30),
            "H_E_boost_0.50": STRATEGIES["E_lgbm_floor_llm_boost"](df, boost=0.50),
        }

        df["_split_id"] = split_id
        for name, sig in runs.items():
            df[f"sig_{name}"] = sig.values
        pooled_dfs.append(df.copy())

        # Evaluate
        target = df["_fwd_r5"]
        split_metrics = {}
        print(f"  {'Method':<28} {'RankIC':>8} {'Top10%':>8} {'WR10':>6} {'Top20%':>8} {'CI95 Top10%':>20}")
        for name, sig in runs.items():
            m = evaluate(sig, target)
            ci = bootstrap_ci(sig, target, n_resample=500)
            m.update(ci)
            split_metrics[name] = m
            print(f"  {name:<28} {m.get('rank_ic', float('nan')):>+8.4f} "
                  f"{m.get('top10_mean', float('nan'))*100:>+7.2f}% "
                  f"{m.get('top10_winrate', float('nan')):>5.1%} "
                  f"{m.get('top20_mean', float('nan'))*100:>+7.2f}% "
                  f"  [{ci.get('top10_ci95', (0,0))[0]*100:+5.2f}%, {ci.get('top10_ci95', (0,0))[1]*100:+5.2f}%]")
        per_split[split_id] = split_metrics

    # Pool all 6000 anchors and re-evaluate
    print("\n=== Pooled across 3 splits (n=6000) ===")
    pooled = pd.concat(pooled_dfs, ignore_index=True)
    print(f"  pooled: {len(pooled)} anchors")

    pooled_metrics = {}
    target_p = pooled["_fwd_r5"]
    print(f"  {'Method':<28} {'RankIC':>8} {'Top10%':>8} {'WR10':>6} {'Top20%':>8} {'CI95 Top10%':>20}")
    sig_cols = [c for c in pooled.columns if c.startswith("sig_")]
    for sig_col in sig_cols:
        name = sig_col[4:]
        m = evaluate(pooled[sig_col], target_p)
        ci = bootstrap_ci(pooled[sig_col], target_p, n_resample=1000)
        m.update(ci)
        pooled_metrics[name] = m
        print(f"  {name:<28} {m.get('rank_ic', float('nan')):>+8.4f} "
              f"{m.get('top10_mean', float('nan'))*100:>+7.2f}% "
              f"{m.get('top10_winrate', float('nan')):>5.1%} "
              f"{m.get('top20_mean', float('nan'))*100:>+7.2f}% "
              f"  [{ci.get('top10_ci95', (0,0))[0]*100:+5.2f}%, {ci.get('top10_ci95', (0,0))[1]*100:+5.2f}%]")

    # Per-split aggregation
    print("\n=== Per-split mean ± std (3 splits) ===")
    methods = list(per_split[1].keys())
    agg = {}
    print(f"  {'Method':<28} {'RankIC':>16} {'Top10%':>16} {'Top20%':>16}")
    for m in methods:
        rics = [per_split[s][m].get("rank_ic", np.nan) for s in [1,2,3]]
        t10s = [per_split[s][m].get("top10_mean", np.nan) for s in [1,2,3]]
        t20s = [per_split[s][m].get("top20_mean", np.nan) for s in [1,2,3]]
        agg[m] = {
            "rank_ic_mean": float(np.nanmean(rics)),
            "rank_ic_std": float(np.nanstd(rics)),
            "top10_mean": float(np.nanmean(t10s)),
            "top10_std": float(np.nanstd(t10s)),
            "top20_mean": float(np.nanmean(t20s)),
            "top20_std": float(np.nanstd(t20s)),
            "per_split": {"rank_ic": rics, "top10": t10s, "top20": t20s},
        }
        print(f"  {m:<28} {np.nanmean(rics):>+7.3f}±{np.nanstd(rics):>5.3f}  "
              f"{np.nanmean(t10s)*100:>+6.2f}±{np.nanstd(t10s)*100:>5.2f}%  "
              f"{np.nanmean(t20s)*100:>+6.2f}±{np.nanstd(t20s)*100:>5.2f}%")

    # H_E vs LGBM: per-split direct comparison
    print("\n=== H_E_boost_0.30 vs BL_LGBM per-split paired comparison ===")
    diff_t10 = []
    for s in [1, 2, 3]:
        h_t10 = per_split[s]["H_E_boost_0.30"]["top10_mean"]
        lgbm_t10 = per_split[s]["BL_LGBM"]["top10_mean"]
        delta = h_t10 - lgbm_t10
        diff_t10.append(delta)
        print(f"  Split {s}: H_E Top10% = {h_t10*100:+.2f}%  vs  LGBM = {lgbm_t10*100:+.2f}%  delta = {delta*100:+.2f}pp")
    print(f"  Mean delta = {np.mean(diff_t10)*100:+.2f}pp, std = {np.std(diff_t10)*100:.2f}pp")
    print(f"  Sign consistency: {sum(1 for d in diff_t10 if d > 0)}/3 splits favor H_E")

    # Save
    pooled.to_parquet(OUT / "pooled_predictions.parquet")
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "per_split": per_split,
            "pooled": pooled_metrics,
            "aggregated": agg,
            "h_e_vs_lgbm_delta_top10": diff_t10,
        }, f, indent=2, default=str)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
