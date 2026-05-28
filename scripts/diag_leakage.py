"""Diagnostic: locate forward-information leakage in D1 features.

Hypothesis: some "feature" column actually contains forward-looking info,
which would explain RankIC = 0.56 (impossible for honest 5-day prediction).

Method: compute Spearman correlation between each feature and forward return,
on a small sample. Any |corr| > 0.5 is a smoking gun.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/ashares_tushare_v1.parquet"


def main():
    print("Loading D1...")
    df = pd.read_parquet(PANEL)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # Forward 5-day return
    g = df.groupby("ts_code")["close"]
    df["_fwd_r5"] = g.shift(-5) / df["close"] - 1
    df["_past_r5"] = df["close"] / g.shift(5) - 1

    # Sanity: 'r5' from factor_lab — is it backward (correlated with past_r5)
    # or forward (correlated with fwd_r5)?
    if "r5" in df.columns:
        sample = df[["r5", "_past_r5", "_fwd_r5"]].dropna().sample(min(50000, len(df)), random_state=42)
        rho_past = spearmanr(sample["r5"], sample["_past_r5"]).correlation
        rho_fwd = spearmanr(sample["r5"], sample["_fwd_r5"]).correlation
        print(f"\n--- 'r5' identity check ---")
        print(f"  spearman(r5, past_r5) = {rho_past:+.4f}")
        print(f"  spearman(r5, fwd_r5)  = {rho_fwd:+.4f}")
        if abs(rho_fwd) > 0.5:
            print(f"  [LEAK] r5 IS FORWARD-LOOKING (leakage!)")
        elif abs(rho_past) > 0.5:
            print(f"  OK r5 is backward (safe)")

    # Same for r20
    if "r20" in df.columns:
        df["_fwd_r20"] = g.shift(-20) / df["close"] - 1
        df["_past_r20"] = df["close"] / g.shift(20) - 1
        sample = df[["r20", "_past_r20", "_fwd_r20"]].dropna().sample(min(50000, len(df)), random_state=42)
        rho_past = spearmanr(sample["r20"], sample["_past_r20"]).correlation
        rho_fwd = spearmanr(sample["r20"], sample["_fwd_r20"]).correlation
        print(f"\n--- 'r20' identity check ---")
        print(f"  spearman(r20, past_r20) = {rho_past:+.4f}")
        print(f"  spearman(r20, fwd_r20)  = {rho_fwd:+.4f}")

    # Full feature scan
    print("\n--- Top 15 features by |corr with fwd_r5| ---")
    NON_FEAT = {"ts_code", "trade_date", "industry", "open", "high", "low", "close",
                "pre_close", "change", "pct_chg", "vol", "amount",
                "_fwd_r5", "_past_r5", "_fwd_r20", "_past_r20"}
    features = [c for c in df.columns if c not in NON_FEAT and pd.api.types.is_numeric_dtype(df[c])]

    # Use a temporal sample (one date) to speed up
    one_date = df[df["trade_date"] == "20240601"]
    if len(one_date) < 100:
        # try another date
        one_date = df[df["trade_date"] == df["trade_date"].iloc[len(df) // 2]]
    print(f"  Sample: {len(one_date)} rows on a single date")

    results = []
    for c in features:
        sub = one_date[[c, "_fwd_r5"]].dropna()
        if len(sub) < 50:
            continue
        rho = spearmanr(sub[c], sub["_fwd_r5"]).correlation
        if rho is not None and not np.isnan(rho):
            results.append((c, rho))

    results.sort(key=lambda x: -abs(x[1]))
    for c, rho in results[:15]:
        flag = " [LEAK] LEAK" if abs(rho) > 0.30 else ""
        print(f"  {c:30s}  rho={rho:+.4f}{flag}")

    # Suspicious factor groups
    print("\n--- Check for 'fwd', 'next', 'future', 'label' substring in column names ---")
    suspicious = [c for c in df.columns if any(s in c.lower() for s in ["fwd", "next", "future", "label", "_y", "ret_5"])]
    if suspicious:
        for c in suspicious:
            print(f"  {c}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
