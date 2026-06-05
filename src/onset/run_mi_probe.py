"""MI2 -- REAL ($0): the decisive marginal-vs-conditional MI probe on A-shares.

We proved the MARGINAL information I(pattern; fwd_r) ~ 0. The promoter-needs-a-
transcription-factor hypothesis says the information is CONDITIONAL: it only
appears once you know the market regime. This script measures, for each candle
pattern feature (and a combined onset score) and each regime variable Z:

  marginal MI   I(feature; fwd_r)            with a GLOBAL permutation null
  conditional MI I(feature; fwd_r | Z)       with a WITHIN-Z permutation null

and reports conditional-minus-marginal plus both permutation p-values. The
go/no-go: does ANY conditional MI clear its permutation null (p<0.05) by a
margin the marginal does not? Targets: raw _fwd_r5 AND the market-neutral
residual (cross-sectional demean per date), so a pure beta effect can't fake it.

SIGN-M1: raw MI is biased -> only permutation-significant conditional MI counts.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_mi_probe
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_tcn_wf import D1, ROOT
from src.onset.candle_geometry import panel_candle_features, FEATURE_COLS
from src.onset.regimes import regime_states, map_states_to_rows
from src.onset.mutual_info import (mutual_info, conditional_mi, perm_pvalue,
                                   interaction_pvalue)

OUT = ROOT / "results/motif"
N_PERM = 300
BINS = 8
SUBSAMPLE = 200_000          # cap rows for the MI estimate (keeps it CPU-cheap)


def _combined_score(feats: pd.DataFrame) -> pd.Series:
    """A simple combined onset score: z-scored sum of the bullish-leaning geometry
    features (breakout, higher_lows, close_loc, vol_ratio, body)."""
    keys = [c for c in ("breakout", "higher_lows", "close_loc", "vol_ratio", "body")
            if c in feats.columns]
    z = (feats[keys] - feats[keys].mean()) / (feats[keys].std() + 1e-9)
    return z.sum(axis=1)


def probe(df: pd.DataFrame, features: list[str], target_cols: list[str],
          n_perm: int = N_PERM, seed: int = 0) -> dict:
    """Marginal + conditional MI (with perm p-values) for each feature x target x
    regime. df must carry the feature columns, the target columns, and integer
    regime columns rg_trend/rg_vol/rg_disaster."""
    regimes = {"trend": "rg_trend", "vol": "rg_vol", "disaster": "rg_disaster"}
    res = {}
    for tgt in target_cols:
        y = df[tgt].to_numpy()
        res[tgt] = {}
        for feat in features:
            x = df[feat].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            xc, yc = x[ok], y[ok]
            entry = {
                "marginal": perm_pvalue(xc, yc, n_perm=n_perm, conditional=False,
                                        bins=BINS, seed=seed)}
            for rname, rcol in regimes.items():
                z = df[rcol].to_numpy()[ok]
                if len(np.unique(z)) < 2:
                    continue
                # cond_ : tests I(X;Y|Z)>0 (trivial if marginal>0; kept for context)
                # interact_ : the DECISIVE motif test -- does the regime ADD info?
                # (Z-permutation null). Only interaction significance counts.
                entry[f"cond_{rname}"] = perm_pvalue(
                    xc, yc, z, n_perm=n_perm, conditional=True, bins=BINS, seed=seed)
                entry[f"interact_{rname}"] = interaction_pvalue(
                    xc, yc, z, n_perm=n_perm, bins=BINS, seed=seed)
            res[tgt][feat] = entry
    return res


def _verdict(res: dict) -> dict:
    """The decisive criterion is the INTERACTION test (Z-permutation), NOT the raw
    conditional permutation. A hit = the regime ADDS information: interaction>0 and
    the Z-permutation p<0.05 (the real regime beats a random regime of equal
    granularity). At n~2e5 the simple permutation p-floors at ~1/n_perm for ANY
    non-zero MI, so we ALSO report the interaction effect size, not just p."""
    hits = []
    for tgt, feats in res.items():
        for feat, e in feats.items():
            for k, v in e.items():
                if not k.startswith("interact_"):
                    continue
                if v["p_value"] < 0.05 and v["interaction"] > 1e-4:
                    hits.append({
                        "target": tgt, "feature": feat, "regime": k,
                        "cond_mi": v["cond_mi"], "marg_mi": v["marg_mi"],
                        "interaction": v["interaction"], "interact_p": v["p_value"],
                        "interaction_frac": (v["interaction"] / v["cond_mi"]
                                             if v["cond_mi"] > 0 else 0.0)})
    hits.sort(key=lambda h: -h["interaction"])
    verdict = ("REGIME-ADDS-INFORMATION (motif worth modeling)" if hits
               else "NO-ADDED-INFORMATION (regime adds nothing; onset info-exhausted)")
    return {"verdict": verdict, "n_hits": len(hits), "hits": hits[:20],
            "criterion": "interaction>0 AND Z-permutation p<0.05"}


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(D1, columns=["ts_code", "trade_date", "open", "high",
                                      "low", "close", "vol", "amount", "pct_chg"])
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["_fwd_r5"] = df.groupby("ts_code")["close"].transform(
        lambda s: s.shift(-5) / s - 1.0)
    # market-neutral residual: subtract the per-date cross-sectional mean
    df["_fwd_r5_neu"] = df["_fwd_r5"] - df.groupby("trade_date")["_fwd_r5"].transform("mean")
    return df


def build(df: pd.DataFrame, subsample: int = SUBSAMPLE, seed: int = 0) -> pd.DataFrame:
    feats = panel_candle_features(df)
    df = pd.concat([df, feats[FEATURE_COLS]], axis=1)
    df["onset_score"] = _combined_score(feats)
    st = regime_states(df)
    df["rg_trend"] = map_states_to_rows(st["trend"], df["trade_date"])
    df["rg_vol"] = map_states_to_rows(st["vol"], df["trade_date"])
    df["rg_disaster"] = map_states_to_rows(st["disaster"], df["trade_date"])
    df = df.dropna(subset=["_fwd_r5"])
    if subsample and len(df) > subsample:
        df = df.sample(subsample, random_state=seed)
    return df


def run_real() -> dict:
    df = build(load_panel())
    features = FEATURE_COLS + ["onset_score"]
    targets = ["_fwd_r5", "_fwd_r5_neu"]
    res = probe(df, features, targets)
    out = {"n_rows": int(len(df)), "n_perm": N_PERM, "bins": BINS,
           "features": features, "targets": targets,
           "results": res, "summary": _verdict(res)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mi_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    out = run_real()
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
