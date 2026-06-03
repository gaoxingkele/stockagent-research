"""ID3 — REAL: the first IDENTIFIED estimate of LLM contribution to onset
prediction, on the leakage-free A-share substrate.

Pipeline (all from existing scored data -> $0 new LLM cost):
  - LLM signals: results/poc_full/predictions.parquet (raw_p_up, expert_p_up)
  - tabular baseline: src/agent/eval_hybrid.add_lgbm_predictions (e1_1 LGBM)
  - leakage validity: results/e3_ashare/summary.json -> ID1 leakage_validity
  - identified contribution: ID2 incremental_contribution (partial rank corr
    beyond baseline) vs the real forward return _fwd_r5, date-clustered CI.

The finding is likely ~0 (consistent with the weak-signal thesis) — and a clean,
identified null is itself the key result: it says the field's positive "LLM
helps" numbers are largely the memorization the A-share data removes.

Run: .venv-xpu\\Scripts\\python.exe -m src.identify.run_ashare_identify
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.identify.contribution import incremental_contribution
from src.identify.leakage_validity import from_summary

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/identify/ashare"


def identify_from_scored(df: pd.DataFrame, baseline_col: str, target_col: str,
                         date_col: str, llm_cols: dict, validity: dict) -> dict:
    """Compute the identified LLM contribution for each named LLM signal.

    df must already contain baseline_col, target_col, date_col and the llm
    signal columns. ``llm_cols`` maps a label -> column name. ``validity`` is the
    ID1 leakage-validity dict for the data.
    """
    sub = df.dropna(subset=[baseline_col, target_col, *llm_cols.values()])
    out = {"leakage_validity": validity, "n": int(len(sub)),
           "identified": "yes" if validity.get("holds") else "NO (data contaminated)",
           "contribution": {}}
    for label, col in llm_cols.items():
        out["contribution"][label] = incremental_contribution(
            sub[baseline_col].values, sub[col].values, sub[target_col].values,
            sub[date_col].astype(str).values)
    return out


def run_real() -> dict:
    from src.agent.eval_hybrid import add_lgbm_predictions
    poc = pd.read_parquet(ROOT / "results/poc_full/predictions.parquet")
    poc = add_lgbm_predictions(poc)                       # adds lgbm_pump_ratio
    validity = from_summary(json.loads((ROOT / "results/e3_ashare/summary.json").read_text(encoding="utf-8")))
    res = identify_from_scored(
        poc, baseline_col="lgbm_pump_ratio", target_col="_fwd_r5", date_col="trade_date",
        llm_cols={"raw": "raw_p_up", "expert": "expert_p_up"}, validity=validity)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    res = run_real()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
