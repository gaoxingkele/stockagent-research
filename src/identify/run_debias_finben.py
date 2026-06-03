"""DB2 — REAL: de-bias the FinBen US stock-movement scores using the A-share
leakage-free calibration (DB1 estimator).

Reads the measured full / no-context accuracies for ACL18, BigData22, CIKM18
(results/e3_*/leakage_summary.json) and the clean A-share calibration
(results/e3_ashare/summary.json + poc_full full-context accuracy), and writes
recall-corrected 'reasoning-only' estimates per benchmark.

$0 new LLM cost (pure reanalysis).

Run: .venv-xpu\\Scripts\\python.exe -m src.identify.run_debias_finben
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.identify.debias import debias_accuracy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/identify/debias"


def debias_finben(benchmarks: dict, clean_full: float, clean_nocontext: float,
                  chance: float = 0.5) -> dict:
    """benchmarks: {name: (full_acc, nocontext_acc)}. Returns per-benchmark
    de-biased estimates."""
    return {name: debias_accuracy(full, noctx, clean_full, clean_nocontext, chance)
            for name, (full, noctx) in benchmarks.items()}


def _ashare_full_acc() -> float:
    """A-share full-context directional accuracy from poc_full (raw_p_up vs sign fwd_r5)."""
    df = pd.read_parquet(ROOT / "results/poc_full/predictions.parquet")[["raw_p_up", "_fwd_r5"]].dropna()
    pred_up = (df["raw_p_up"] >= 0.5).astype(int)
    true_up = (df["_fwd_r5"] > 0).astype(int)
    return float((pred_up == true_up).mean())


def run_real() -> dict:
    benchmarks = {}
    for ds in ("acl", "bigdata", "cikm"):
        s = json.loads((ROOT / f"results/e3_{ds}/leakage_summary.json").read_text(encoding="utf-8"))
        benchmarks[ds] = (s["llm_full_context"]["acc"], s["llm_no_context"]["acc"])
    ash = json.loads((ROOT / "results/e3_ashare/summary.json").read_text(encoding="utf-8"))
    clean_nocontext = ash["no_context_accuracy"]
    clean_full = _ashare_full_acc()
    out = {"clean_market": "A-shares", "clean_full": clean_full,
           "clean_nocontext": clean_nocontext,
           "corrected": debias_finben(benchmarks, clean_full, clean_nocontext)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "finben_corrected.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
