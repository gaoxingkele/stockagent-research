"""TRD2 -- REAL ($0): the decisive regime-gated long-only cost-aware backtest.

TRD1 found the conditional mean is monotone in the candle features within the
trend regime (a real directional core, ~60% sign-blind). This is the money test:
build the SIMPLEST regime-gated long-only top-K portfolio on the best features,
net of realistic A-share round-trip cost, market-excess, on NON-OVERLAPPING 5-day
periods (no autocorrelation inflation), with date-clustered bootstrap CIs, pooled
AND per-year (2022-2025). Also compares gated vs always-on (ungated) to isolate
the regime's contribution. Does the monotone directional core survive into
net-of-cost return that holds cross-period? If not -> information-only.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.run_regime_gate_bt
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.train_tcn_wf import D1
from src.onset.candle_geometry import panel_candle_features, FEATURE_COLS
from src.onset.regimes import trend_state
from src.onset.regime_gate import regime_gated_excess, gated_vs_ungated
from src.onset.long_only import summarize_excess
from src.onset.ashare_cost import DEFAULT_ROUND_TRIP
from src.onset.run_mi_probe import OUT, _combined_score

FEATURES = ["close_pct_prior", "close_loc", "onset_score"]
K_FRAC = 0.1
YEARS = ("2022", "2023", "2024", "2025")


def load_features() -> tuple:
    df = pd.read_parquet(D1, columns=["ts_code", "trade_date", "open", "high",
                                      "low", "close", "vol", "amount", "pct_chg"])
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["_fwd_r5"] = df.groupby("ts_code")["close"].transform(
        lambda s: s.shift(-5) / s - 1.0)
    feats = panel_candle_features(df)
    df = pd.concat([df, feats[FEATURE_COLS]], axis=1)
    df["onset_score"] = _combined_score(feats)
    # NON-OVERLAPPING 5-day periods: keep every 5th trade_date
    all_dates = sorted(df["trade_date"].unique())[::5]
    df = df[df["trade_date"].isin(all_dates)]
    market_ret = df.groupby("trade_date")["_fwd_r5"].mean().sort_index()
    in_regime = trend_state(market_ret).astype(bool)   # trend-up = activating
    return df, in_regime


def _bt_one(df: pd.DataFrame, feature: str, in_regime: pd.Series) -> dict:
    gv = gated_vs_ungated(df, feature, in_regime, k_frac=K_FRAC,
                          cost=DEFAULT_ROUND_TRIP)
    gated, ungated = gv["gated"], gv["ungated"]
    out = {"in_regime_frac": gv["in_regime_frac"],
           "pooled_gated": summarize_excess(gated, n_boot=1000),
           "pooled_ungated": summarize_excess(ungated, n_boot=1000),
           "per_year": {}}
    pos_years = 0
    for yr in YEARS:
        idx = [d for d in gated.index if d[:4] == yr]
        if len(idx) >= 5:
            sy = summarize_excess(gated.reindex(idx), n_boot=500)
            out["per_year"][yr] = sy
            ci = sy.get("mean_ci95")
            if ci and ci[0] > 0:
                pos_years += 1
    out["n_pos_years"] = pos_years
    out["n_years"] = len(out["per_year"])
    return out


def run_real() -> dict:
    df, in_regime = load_features()
    results = {f: _bt_one(df, f, in_regime) for f in FEATURES}
    # decisive: any feature whose gated net excess CI>0 in >=2/3 years
    survivors = [f for f, r in results.items()
                 if r["n_years"] >= 2 and r["n_pos_years"] >= max(2, int(np.ceil(2 * r["n_years"] / 3)))]
    verdict = ("NET-OF-COST-TRADABLE" if survivors else
               "EATEN-BY-COST-OR-NONSTATIONARY (information-only)")
    out = {"k_frac": K_FRAC, "round_trip_cost": DEFAULT_ROUND_TRIP,
           "features": FEATURES, "results": results,
           "survivors": survivors, "verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "regime_gate_bt.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    out = run_real()
    for f, r in out["results"].items():
        pg = r["pooled_gated"]; pu = r["pooled_ungated"]
        print(f"{f:<16} gated Sharpe={pg.get('annualized_sharpe'):+.2f} "
              f"CI={pg.get('mean_ci95')} | ungated Sharpe={pu.get('annualized_sharpe'):+.2f} "
              f"| pos_years={r['n_pos_years']}/{r['n_years']} regime_frac={r['in_regime_frac']:.2f}")
    print("SURVIVORS:", out["survivors"], "| VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
