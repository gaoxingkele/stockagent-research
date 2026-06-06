"""EXP-B -- REAL ($0): two-level-uncertainty abstention on the onset ranker.

Does When-Alpha-Breaks (2603.13252) abstention rescue the sub-cost onset edge
(TRD2)? Three arms over the same non-overlapping 5-day A-share grid, long-only
top-decile by the onset score, net of round-trip cost, date-clustered CI + DSR,
per-year 2022-2025:
  always   : trade every period
  trend    : trade only in our trend-up regime
  abstain  : trade only when BOTH model uncertainty AND regime instability are
             below their expanding-past quantiles (BENCH3)

Model uncertainty proxy = negative cross-sectional dispersion of the onset score
that date (low spread -> low ranking confidence -> high uncertainty), point-in-time.
Regime instability = trailing trend-state switch rate.

Run: .venv-xpu\\Scripts\\python.exe -m src.bench.run_expb
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.train_tcn_wf import ROOT
from src.onset.long_only import summarize_excess
from src.onset.ashare_cost import DEFAULT_ROUND_TRIP
from src.onset.regime_gate import regime_gated_excess
from src.onset.regimes import trend_state
from src.bench.abstention import regime_instability, abstain_mask
from src.bench.run_expa import arm_metrics
from src.onset.run_regime_gate_bt import load_features, K_FRAC

OUT = ROOT / "results/bench"
N_ARMS = 3
YEARS = ("2022", "2023", "2024", "2025")
SIGNAL = "onset_score"


def _per_date_uncertainty(df: pd.DataFrame, dates: list) -> np.ndarray:
    disp = df.groupby("trade_date")[SIGNAL].std().reindex(dates).ffill().bfill()
    return (-disp).to_numpy()          # low dispersion -> high uncertainty


def evaluate(df: pd.DataFrame, trend_up: pd.Series) -> dict:
    dates = sorted(df["trade_date"].unique())
    market_ret = df.groupby("trade_date")[SIGNAL].mean().reindex(dates)  # grid only
    tstate = trend_up.reindex(dates).fillna(True).astype(int).to_numpy()
    instab = regime_instability(tstate, window=4)
    unc = _per_date_uncertainty(df, dates)
    trade = abstain_mask(unc, instab, q_unc=0.8, q_reg=0.8, min_warmup=20)
    abstain_gate = pd.Series(trade, index=dates)

    gates = {"always": pd.Series(True, index=dates),
             "trend": trend_up.reindex(dates).fillna(True).astype(bool),
             "abstain": abstain_gate}
    pooled_series, per_year = {}, {k: {} for k in gates}
    for name, gate in gates.items():
        ex = regime_gated_excess(df, SIGNAL, gate, ret="_fwd_r5",
                                 k_frac=K_FRAC, cost=DEFAULT_ROUND_TRIP)
        pooled_series[name] = ex
        for yr in YEARS:
            idx = [d for d in ex.index if d[:4] == yr]
            if len(idx) >= 5:
                per_year[name][yr] = arm_metrics(ex.reindex(idx), n_trials=N_ARMS)
    sr_list = [summarize_excess(s, n_boot=200).get("annualized_sharpe", float("nan"))
               for s in pooled_series.values()]
    pooled = {k: arm_metrics(s, n_trials=N_ARMS, trial_sharpes=sr_list)
              for k, s in pooled_series.items()}
    traded_frac = {k: float((g.reindex(dates).fillna(False)).mean()) for k, g in gates.items()}
    return {"pooled": pooled, "per_year": per_year, "traded_frac": traded_frac}


def _verdict(res: dict) -> dict:
    ab = res["pooled"].get("abstain", {})
    ci = ab.get("mean_ci95"); dsr = ab.get("deflated", {}).get("dsr", 0.0)
    rescued = bool(ci and ci[0] > 0 and dsr > 0.95)
    return {"abstention_rescues": rescued,
            "verdict": ("ABSTENTION RESCUES (net-of-cost positive)" if rescued
                        else "ABSTENTION DOES NOT RESCUE (onset edge stays sub-cost)")}


def run_real() -> dict:
    df, trend_up = load_features()
    res = evaluate(df, trend_up)
    res["summary"] = _verdict(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "expb.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    res = run_real()
    for arm, m in res["pooled"].items():
        d = m.get("deflated", {})
        print(f"{arm:<8} Sharpe={m.get('annualized_sharpe'):+.2f} DSR={d.get('dsr'):.3f} "
              f"CI={m.get('mean_ci95')} traded={res['traded_frac'][arm]:.2f}")
    print("VERDICT:", res["summary"]["verdict"])


if __name__ == "__main__":
    main()
