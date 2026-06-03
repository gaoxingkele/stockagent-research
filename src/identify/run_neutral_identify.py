"""NB3 — REAL ($0): market/sector-neutral identification + tradable long-short
on the leakage-free A-share substrate.

Answers, on existing scored data ($0):
  - Is the idiosyncratic ALPHA contribution of the LLM (over the LGBM baseline)
    null once we neutralise market/sector beta?  (selection, on the neutral target)
  - Does the LLM contribute to market TIMING (beta)?  (timing)
  - Is there ANY tradable market-neutral signal in the baseline?  (long-short Sharpe)

Run: .venv-xpu\\Scripts\\python.exe -m src.identify.run_neutral_identify
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.identify.neutral_targets import decompose
from src.identify.decompose_identify import selection_contribution, timing_contribution
from src.identify.leakage_validity import from_summary

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/identify/neutral"
PERIODS_PER_YEAR = 252 / 5.0          # _fwd_r5 is a 5-trading-day return


def long_short(df: pd.DataFrame, sig_col: str, ret: str = "_fwd_r5",
               date: str = "trade_date", frac: float = 0.2,
               n_boot: int = 1000, seed: int = 42) -> dict:
    """Dollar-neutral (market-neutral) top-minus-bottom long-short of `ret`,
    ranked by `sig_col` each date. Returns per-period mean, annualised Sharpe,
    and a date-block-bootstrap 95% CI on the mean."""
    def per_date_ls(sub):
        s = sub.dropna(subset=[sig_col, ret])
        if len(s) < 5:
            return np.nan
        k = max(1, int(len(s) * frac))
        order = s.sort_values(sig_col)
        return order[ret].iloc[-k:].mean() - order[ret].iloc[:k].mean()

    ls = df.groupby(date).apply(per_date_ls).dropna()
    if len(ls) < 2:
        return {"n_dates": int(len(ls))}
    mean = float(ls.mean()); sd = float(ls.std())
    sharpe = float(np.sqrt(PERIODS_PER_YEAR) * mean / sd) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    vals = ls.values; m = len(vals)
    boot = [vals[rng.integers(0, m, m)].mean() for _ in range(n_boot)]
    return {"ls_mean_per_period": mean, "annualized_sharpe": sharpe, "n_dates": int(m),
            "ls_mean_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def neutral_identify(df: pd.DataFrame, baseline_col: str, llm_cols: dict, validity: dict) -> dict:
    d = decompose(df)
    dates = d["trade_date"].astype(str).values
    out = {"leakage_validity": validity,
           "identified": "yes" if validity.get("holds") else "NO (contaminated)",
           "n": int(len(d)), "selection": {}, "timing": {}}
    for tgt_name, tgt_col, syst_col in [("market_neutral", "mkt_neutral", "mkt_systematic"),
                                        ("sector_neutral", "sec_neutral", "sec_systematic")]:
        if tgt_col not in d.columns:
            continue
        out["selection"][tgt_name] = {}
        out["timing"][tgt_name] = {}
        for label, col in llm_cols.items():
            sub = d.dropna(subset=[baseline_col, col, tgt_col, syst_col])
            out["selection"][tgt_name][label] = selection_contribution(
                sub[baseline_col].values, sub[col].values, sub[tgt_col].values,
                sub["trade_date"].astype(str).values)
            out["timing"][tgt_name][label] = timing_contribution(
                sub[col].values, sub[syst_col].values, sub["trade_date"].astype(str).values)
    # tradable long-short of the baseline and of the raw LLM
    out["long_short"] = {
        "baseline": long_short(d, baseline_col),
        "llm_" + next(iter(llm_cols)): long_short(d, next(iter(llm_cols.values()))),
    }
    return out


def run_real() -> dict:
    from src.agent.eval_hybrid import add_lgbm_predictions
    poc = pd.read_parquet(ROOT / "results/poc_full/predictions.parquet")
    poc = add_lgbm_predictions(poc)
    validity = from_summary(json.loads((ROOT / "results/e3_ashare/summary.json").read_text(encoding="utf-8")))
    res = neutral_identify(poc, "lgbm_pump_ratio",
                           {"raw": "raw_p_up", "expert": "expert_p_up"}, validity)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
