"""Evaluate regime-aware routers (F/G/H/I) + oracle upper bound + W1.5 baselines."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

from src.agent.hybrid_router import STRATEGIES, _rankify

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/wf_regime_eval"
OUT.mkdir(parents=True, exist_ok=True)
D1 = ROOT / "data/processed/ashares_tushare_v1.parquet"

NON_FEAT = {
    "ts_code", "trade_date", "industry",
    "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount",
    "r5", "r10", "r20", "r30", "r40",
    "dd5", "dd10", "dd20", "dd30", "dd40",
}

SPLITS = {1: "poc_wf_split1", 2: "poc_wf_split2", 3: "poc_wf_split3"}


def add_lgbm(df, model_path, d1):
    booster = lgb.Booster(model_file=str(model_path))
    key = df[["ts_code", "trade_date"]]
    d1_anchored = key.merge(d1, on=["ts_code", "trade_date"], how="left")
    feat_cols = [c for c in d1_anchored.columns
                  if c not in NON_FEAT
                  and pd.api.types.is_numeric_dtype(d1_anchored[c])
                  and d1_anchored[c].notna().mean() >= 0.05]
    X = d1_anchored[feat_cols]
    prob = booster.predict(X, num_iteration=booster.best_iteration)
    df = df.copy()
    df["lgbm_p_down"] = prob[:, 0]
    df["lgbm_p_neutral"] = prob[:, 1]
    df["lgbm_p_up"] = prob[:, 2]
    df["lgbm_pump_ratio"] = prob[:, 2] / (prob[:, 0] + 0.01)
    df["lgbm_max_prob"] = prob.max(axis=1)
    return df


def add_market_context(df, d1):
    """Attach _mkt_is_disaster_month and _mkt_signal_A/B if not present."""
    if "_mkt_is_disaster_month" in df.columns:
        return df
    from src.onset.disaster_filter import compute_daily_market_signals, compute_disaster_signals
    m = compute_daily_market_signals(
        d1[["ts_code", "trade_date", "pct_chg", "amount", "industry", "total_mv"]]
    )
    d = compute_disaster_signals(m)
    market = m.join(d[["signal_A_index", "signal_B_volume", "signal_C_sector",
                        "outer_vote_count", "is_disaster_month"]])
    df = df.copy()
    df["_mkt_is_disaster_month"] = df["trade_date"].map(market["is_disaster_month"])
    df["_mkt_signal_A_index"] = df["trade_date"].map(market["signal_A_index"])
    df["_mkt_signal_B_volume"] = df["trade_date"].map(market["signal_B_volume"])
    return df


def evaluate(signal, target):
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
        "top10_mean": float(top10["tgt"].mean()),
        "top10_winrate": float((top10["tgt"] > 0).mean()),
        "top20_mean": float(top20["tgt"].mean()),
    }


def bootstrap_ci(signal, target, n_resample=1000, seed=42):
    rng = np.random.default_rng(seed)
    sub = pd.DataFrame({"sig": signal, "tgt": target}).dropna()
    if len(sub) < 100:
        return {}
    n = len(sub)
    t10s, t20s = [], []
    for _ in range(n_resample):
        idx = rng.integers(0, n, size=n)
        sample = sub.iloc[idx]
        sorted_s = sample.sort_values("sig", ascending=False)
        t10s.append(sorted_s.head(max(1, n // 10))["tgt"].mean())
        t20s.append(sorted_s.head(max(1, n // 5))["tgt"].mean())
    return {
        "top10_ci95": (float(np.percentile(t10s, 2.5)),
                       float(np.percentile(t10s, 97.5))),
        "top20_ci95": (float(np.percentile(t20s, 2.5)),
                       float(np.percentile(t20s, 97.5))),
    }


def main():
    print("Loading D1...")
    d1 = pd.read_parquet(D1)
    d1["trade_date"] = d1["trade_date"].astype(str)

    pooled = []
    per_split = {}

    for split_id, dir_name in SPLITS.items():
        print(f"\n=== Split {split_id} ===")
        df = pd.read_parquet(ROOT / f"results/{dir_name}/predictions.parquet")
        df = add_lgbm(df, ROOT / f"results/wf_lgbm_split{split_id}/model.txt", d1)
        df = add_market_context(df, d1)
        print(f"  loaded {len(df)} anchors")
        print(f"  disaster days within split: "
              f"{df['_mkt_is_disaster_month'].fillna(False).astype(bool).sum()} anchors")

        runs = {
            "BL_LGBM": df["lgbm_pump_ratio"],
            "BL_LLM_raw": df["raw_p_up"],
            "BL_LLM_expert": df["expert_p_up"],
            "H_E_boost_0.15": STRATEGIES["E_lgbm_floor_llm_boost"](df, boost=0.15),
            "H_E_boost_0.30": STRATEGIES["E_lgbm_floor_llm_boost"](df, boost=0.30),
            "F_disaster_aware": STRATEGIES["F_disaster_aware"](df),
            "G_disagreement_boost": STRATEGIES["G_disagreement_boost"](df),
            "H_market_regime_ensemble": STRATEGIES["H_market_regime_ensemble"](df),
            "I_onset_aware_boost": STRATEGIES["I_onset_aware_boost"](df),
        }
        df["_split_id"] = split_id
        for n, s in runs.items():
            df[f"sig_{n}"] = s.values
        pooled.append(df.copy())

        target = df["_fwd_r5"]
        split_m = {}
        print(f"  {'Method':<28} {'RankIC':>8} {'Top10%':>8} {'WR10':>6} {'Top20%':>8}")
        for n, s in runs.items():
            m = evaluate(s, target)
            split_m[n] = m
            print(f"  {n:<28} {m.get('rank_ic', float('nan')):>+8.4f} "
                  f"{m.get('top10_mean', 0)*100:>+7.2f}% "
                  f"{m.get('top10_winrate', 0):>5.1%} "
                  f"{m.get('top20_mean', 0)*100:>+7.2f}%")
        per_split[split_id] = split_m

    pooled = pd.concat(pooled, ignore_index=True)
    print(f"\n=== POOLED (n={len(pooled)}) ===")
    target = pooled["_fwd_r5"]
    pooled_m = {}
    print(f"  {'Method':<28} {'RankIC':>8} {'Top10%':>8} {'WR10':>6} {'Top20%':>8} {'CI95 Top10%':>20}")
    for col in [c for c in pooled.columns if c.startswith("sig_")]:
        n = col[4:]
        m = evaluate(pooled[col], target)
        ci = bootstrap_ci(pooled[col], target, n_resample=1000)
        m.update(ci)
        pooled_m[n] = m
        print(f"  {n:<28} {m.get('rank_ic', float('nan')):>+8.4f} "
              f"{m.get('top10_mean', 0)*100:>+7.2f}% "
              f"{m.get('top10_winrate', 0):>5.1%} "
              f"{m.get('top20_mean', 0)*100:>+7.2f}% "
              f"  [{ci.get('top10_ci95', (0,0))[0]*100:+5.2f}%, {ci.get('top10_ci95', (0,0))[1]*100:+5.2f}%]")

    # Oracle (best-per-split, upper bound only)
    print("\n=== Oracle upper bound (pick best method per split, retrospective) ===")
    oracle_t10 = []
    for s in [1, 2, 3]:
        methods = list(per_split[s].keys())
        best = max(methods, key=lambda m: per_split[s][m].get("top10_mean", -1))
        oracle_t10.append(per_split[s][best]["top10_mean"])
        print(f"  Split {s} oracle: {best} Top10%={per_split[s][best]['top10_mean']*100:+.2f}%")
    print(f"  Mean oracle Top10% = {np.mean(oracle_t10)*100:+.2f}%")
    lgbm_t10 = [per_split[s]["BL_LGBM"]["top10_mean"] for s in [1,2,3]]
    print(f"  Mean LGBM Top10%   = {np.mean(lgbm_t10)*100:+.2f}%")
    print(f"  Oracle - LGBM = {(np.mean(oracle_t10) - np.mean(lgbm_t10))*100:+.2f}pp (upper bound for any router)")

    # Per-split aggregation
    print("\n=== Per-split aggregation (mean ± std) ===")
    methods = list(per_split[1].keys())
    agg = {}
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
        }
        print(f"  {m:<28} RankIC {np.nanmean(rics):>+.3f}±{np.nanstd(rics):.3f}  "
              f"Top10% {np.nanmean(t10s)*100:>+5.2f}±{np.nanstd(t10s)*100:.2f}%  "
              f"Top20% {np.nanmean(t20s)*100:>+5.2f}±{np.nanstd(t20s)*100:.2f}%")

    pooled.to_parquet(OUT / "pooled_predictions.parquet")
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"per_split": per_split, "pooled": pooled_m,
                    "aggregated": agg,
                    "oracle_top10": oracle_t10,
                    "lgbm_top10": lgbm_t10}, f, indent=2, default=str)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
