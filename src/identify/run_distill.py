"""WS2 — REAL: LLM-as-weak-supervisor distillation with clean attribution.

Question: does LLM *knowledge* (not memory) improve a downstream onset learner?
On the leakage-free A-share substrate (ID1 holds) any held-out gain is cleanly
attributable to knowledge, not recall.

Design (two arms, same features, walk-forward):
  Arm A (no LLM):    downstream learner trained on the true up/down labels.
  Arm B (+LLM weak): training labels REFINED by LLM weak supervision -- where the
                     LLM labeling-functions (WS1) aggregated by the T-002 label
                     model are confident, the label is replaced by the LLM's;
                     otherwise the true label is kept.
Held-out (split3) clustered RankIC of predicted P(up) vs _fwd_r5 for each arm.

Engineering note (SIGN-R1/002): the downstream learner is LightGBM (fast,
deterministic, CPU) rather than the heavy TCN of train_onset_real -- the clean
attribution question does not need the encoder, and a hermetic/fast learner
keeps the gate meaningful. Reuses src/onset/weak_supervision + src/identify/
llm_lf + src/evaluation/onset_eval.

$0 new LLM cost (reuses poc_wf scored anchors).

Run: .venv-xpu\\Scripts\\python.exe -m src.identify.run_distill
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.identify.llm_lf import llm_to_lf
from src.onset.weak_supervision import label_model
from src.evaluation.onset_eval import clustered_bootstrap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/identify/distill"

_EXCLUDE_PREFIX = ("_", "raw_", "expert_", "lgbm_", "sig_", "tcn_")
_EXCLUDE = {"ts_code", "trade_date", "industry", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "vol", "amount"}


def feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if not c.startswith(_EXCLUDE_PREFIX) and c not in _EXCLUDE
            and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().mean() >= 0.5]


def _rank_ic(pred, target):
    pr = np.argsort(np.argsort(pred)); tr = np.argsort(np.argsort(target))
    return float(np.corrcoef(pr, tr)[0, 1])


def _fit_predict(tr_X, tr_y, te_X):
    d = lgb.Dataset(tr_X, label=tr_y)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15,
                  min_data_in_leaf=30, verbose=-1, seed=42)
    booster = lgb.train(params, d, num_boost_round=80)
    return booster.predict(te_X)


def distill_arms(train_df: pd.DataFrame, test_df: pd.DataFrame, feats: list,
                 conf_band: float = 0.3) -> dict:
    tr_X, te_X = train_df[feats].values, test_df[feats].values
    true_y = (train_df["_fwd_r5"].values > 0).astype(int)

    # Arm A: true labels
    pred_a = _fit_predict(tr_X, true_y, te_X)

    # Arm B: LLM-weak-refined labels
    lf = llm_to_lf(train_df)
    soft, _ = label_model(lf)
    confident = np.abs(soft - 0.5) >= conf_band
    refined_y = true_y.copy()
    refined_y[confident] = (soft[confident] >= 0.5).astype(int)
    pred_b = _fit_predict(tr_X, refined_y, te_X)

    tgt = test_df["_fwd_r5"].values
    dates = test_df["trade_date"].astype(str).values
    ic_a = clustered_bootstrap(_rank_ic, pred_a, tgt, dates, n_boot=300)
    ic_b = clustered_bootstrap(_rank_ic, pred_b, tgt, dates, n_boot=300)
    return {
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "n_features": len(feats), "frac_labels_refined_by_llm": float(confident.mean()),
        "arm_A_true_labels": ic_a,
        "arm_B_llm_weak_refined": ic_b,
        "identified_improvement": ic_b["mean"] - ic_a["mean"],
    }


def run_real() -> dict:
    # poc_wf predictions hold LLM scores + labels but NOT the technical features;
    # merge them from D1 by (ts_code, trade_date), as add_lgbm does.
    from src.train_tcn_wf import D1, select_feature_columns
    d1 = pd.read_parquet(D1)
    d1["trade_date"] = d1["trade_date"].astype(str)
    feats = select_feature_columns(d1)
    keep = ["ts_code", "trade_date", "_fwd_r5", "raw_p_up", "raw_pump_ratio", "_exp_onset_score"]

    def load(s):
        p = pd.read_parquet(ROOT / f"results/poc_wf_split{s}/predictions.parquet")
        p["trade_date"] = p["trade_date"].astype(str)
        return p[keep].merge(d1[["ts_code", "trade_date"] + feats], on=["ts_code", "trade_date"], how="left")

    parts = {s: load(s) for s in (1, 2, 3)}
    train_df = pd.concat([parts[1], parts[2]], ignore_index=True).dropna(subset=["_fwd_r5"] + feats)
    test_df = parts[3].dropna(subset=["_fwd_r5"] + feats)
    res = distill_arms(train_df, test_df, feats)
    res["note"] = "leakage-free (A-share, ID1 holds); improvement is clean LLM-knowledge attribution"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
