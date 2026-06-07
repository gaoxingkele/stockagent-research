"""EXP-A -- REAL ($0): Regime-Aware LightGBM vs our regime-gating vs plain LGBM.

Head-to-head with the methodological twin (Regime-Aware LightGBM, MDPI Electronics
2026), reimplemented on OUR data + OUR honest protocol (SIGN-B1): same D1, same
walk-forward (SPLITS test windows 2025Q2/Q3/Q4), same A-share round-trip cost,
non-overlapping 5-day periods, date-clustered CIs, and the Deflated Sharpe Ratio.

All three arms share ONE LightGBM cross-sectional ranker and differ ONLY in the
GATE, isolating the regime contribution:
  (a) plain    : trade every period (no gate)
  (b) hmm      : trade only in the favorable rolling-HMM regime (BENCH1)
  (c) trend    : trade only in our trend-up regime (regimes.trend_state)

Run: .venv-xpu\\Scripts\\python.exe -m src.bench.run_expa
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import skew as _skew, kurtosis as _kurt

from src.train_tcn_wf import D1, ROOT, SPLITS, select_feature_columns
from src.onset.long_only import summarize_excess, PERIODS_PER_YEAR
from src.onset.ashare_cost import DEFAULT_ROUND_TRIP
from src.onset.regime_gate import regime_gated_excess
from src.onset.regimes import trend_state
from src.bench.hmm_regime import rolling_hmm_states
from src.bench.deflated_sharpe import deflated_sharpe

OUT = ROOT / "results/bench"
K_FRAC = 0.1
N_ARMS = 3                      # multiple-testing count for the DSR (SIGN-B1)


def arm_metrics(excess: pd.Series, n_trials: int, trial_sharpes=None) -> dict:
    s = summarize_excess(excess, n_boot=800)
    sr = s.get("annualized_sharpe", float("nan"))
    e = excess.dropna().to_numpy()
    sk = float(_skew(e)) if len(e) > 2 else 0.0
    ku = float(_kurt(e, fisher=False)) if len(e) > 2 else 3.0
    s["skew"], s["kurtosis"] = sk, ku
    s["deflated"] = deflated_sharpe(sr, n_obs=len(e), skew=sk, kurtosis=ku,
                                    n_trials=n_trials, trial_sharpes=trial_sharpes,
                                    periods_per_year=PERIODS_PER_YEAR)
    return s


def _favorable_hmm(market_ret: pd.Series, n_states: int = 3) -> pd.Series:
    """Favorable = top (highest-return) canonical HMM state, on [ret, rolling vol]."""
    vol = market_ret.rolling(8, min_periods=2).std().fillna(0.0)
    feats = np.column_stack([market_ret.to_numpy(), vol.to_numpy()])
    st = rolling_hmm_states(feats, n_states=n_states, min_train=120, refit_every=40)
    return pd.Series(st, index=market_ret.index) == (n_states - 1)


def load() -> pd.DataFrame:
    df = pd.read_parquet(D1)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["_fwd_r5"] = df.groupby("ts_code")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
    return df


def _nonoverlap(dates) -> list:
    return sorted(set(dates))[::5]


def evaluate(df: pd.DataFrame, n_estimators: int = 300) -> dict:
    import lightgbm as lgb
    feat = [c for c in select_feature_columns(df) if not c.startswith("_fwd")]
    market_ret = df.groupby("trade_date")["_fwd_r5"].mean().sort_index()
    fav_hmm = _favorable_hmm(market_ret)
    trend_up = trend_state(market_ret).astype(bool)

    arms = {"plain": [], "hmm": [], "trend": []}    # per-split net excess series
    per_split = {"plain": {}, "hmm": {}, "trend": {}}
    for sp, (start, train_end, val_end, test_end) in SPLITS.items():
        tr = df[(df["trade_date"] >= start) & (df["trade_date"] < val_end)].dropna(subset=["_fwd_r5"])
        te = df[(df["trade_date"] >= val_end) & (df["trade_date"] < test_end)].copy()
        if len(tr) < 1000 or te.empty:
            continue
        model = lgb.LGBMRegressor(n_estimators=n_estimators, num_leaves=31,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, n_jobs=-1, verbose=-1)
        model.fit(tr[feat], tr["_fwd_r5"])
        te["pred"] = model.predict(te[feat])
        te = te.dropna(subset=["_fwd_r5"])
        keep = set(_nonoverlap(te["trade_date"].unique()))
        te = te[te["trade_date"].isin(keep)]
        alld = pd.Series(True, index=sorted(te["trade_date"].unique()))
        gates = {"plain": alld, "hmm": fav_hmm, "trend": trend_up}
        for name, gate in gates.items():
            ex = regime_gated_excess(te, "pred", gate, ret="_fwd_r5",
                                     k_frac=K_FRAC, cost=DEFAULT_ROUND_TRIP)
            arms[name].append(ex)
            yr = f"{val_end[:4]}Q{(int(val_end[4:6]) - 1) // 3 + 1}"
            per_split[name][yr] = arm_metrics(ex, n_trials=N_ARMS)
    # pooled
    pooled_series = {k: pd.concat(v) for k, v in arms.items() if v}
    pooled_sr = {k: summarize_excess(s, n_boot=200).get("annualized_sharpe", float("nan"))
                 for k, s in pooled_series.items()}
    pooled = {k: arm_metrics(s, n_trials=N_ARMS, trial_sharpes=list(pooled_sr.values()))
              for k, s in pooled_series.items()}
    return {"k_frac": K_FRAC, "cost": DEFAULT_ROUND_TRIP, "n_arms": N_ARMS,
            "pooled": pooled, "per_split": per_split}


def _verdict(res: dict) -> dict:
    survivors = []
    for arm, m in res["pooled"].items():
        ci = m.get("mean_ci95")
        dsr = m.get("deflated", {}).get("dsr", 0.0)
        if ci and ci[0] > 0 and dsr > 0.95:
            survivors.append(arm)
    return {"survivors": survivors,
            "verdict": ("ARM SURVIVES: " + ",".join(survivors)) if survivors
            else "ALL SUB-COST / NOT DSR-SIGNIFICANT (regime baselines do not survive our protocol)"}


def run_real(n_estimators: int = 300) -> dict:
    df = load()
    res = evaluate(df, n_estimators=n_estimators)
    res["summary"] = _verdict(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "expa.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    res = run_real()
    for arm, m in res["pooled"].items():
        d = m.get("deflated", {})
        print(f"{arm:<7} Sharpe={m.get('annualized_sharpe'):+.2f} "
              f"DSR={d.get('dsr'):.3f} CI={m.get('mean_ci95')}")
    print("VERDICT:", res["summary"]["verdict"])


if __name__ == "__main__":
    main()
