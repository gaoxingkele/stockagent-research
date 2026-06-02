"""C5 causal control — no-context LLM probe on A-shares (the deployment market).

FinBen's stock-movement benchmarks (ACL18/BigData22/CIKM18) are entirely
PRE-training-cutoff, so a temporal pre/post split is impossible. Instead we run
the SAME no-context protocol (ticker + date only, no prices/news) on Chinese
A-shares — a market the LLM has barely memorized — and contrast with the US
benchmarks (72-80% no-context). If A-shares come out at chance, the apparent US
"skill" is provably memorization of US data, not forecasting; and the paper's
choice of A-share 2025 walk-forward data is validated as leakage-resistant.

Label: Rise if forward 5-day return > 0 else Fall (directional accuracy vs
sign(_fwd_r5)), matching the FinBen Rise/Fall convention.

Usage: .venv-xpu\\Scripts\\python.exe -m src.eval_e3.ashare_probe
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent.llm_client import call_llm
from src.eval_e3.run_llm import SYSTEM, parse
from src.evaluation.onset_eval import clustered_bootstrap

ROOT = Path(__file__).resolve().parents[2]
ANCHORS = ROOT / "results/poc_full/predictions.parquet"
OUT = ROOT / "results/e3_ashare"
MODEL = "gemini-3.5-flash"
_lock = threading.Lock()


def _prompt(ts_code: str, date: str) -> str:
    return (f"Will the Chinese A-share stock with code {ts_code} go upwards or "
            f"downwards at {date}? Please indicate either Rise or Fall.")


def _score_one(row):
    try:
        r = call_llm(system=SYSTEM, user=_prompt(row["ts_code"], str(row["trade_date"])),
                     model=MODEL, temperature=0.0, max_tokens=256)
        direction, _ = parse(r.content)
        return {"ts_code": row["ts_code"], "trade_date": str(row["trade_date"]),
                "llm_direction": direction, "fwd_r5": float(row["_fwd_r5"]),
                "cost_usd": r.cost_usd}
    except Exception as e:
        return {"ts_code": row["ts_code"], "trade_date": str(row["trade_date"]),
                "llm_direction": None, "fwd_r5": float(row["_fwd_r5"]),
                "cost_usd": 0.0, "error": str(e)[:120]}


def main():
    df = pd.read_parquet(ANCHORS)[["ts_code", "trade_date", "_fwd_r5"]].dropna()
    OUT.mkdir(parents=True, exist_ok=True)

    results, spend = [], 0.0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_score_one, row) for _, row in df.iterrows()]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            spend += res.get("cost_usd", 0.0)
            if i % 200 == 0:
                with _lock:
                    pd.DataFrame(results).to_parquet(OUT / "probe.parquet", index=False)
                print(f"  {i}/{len(df)} | spend ${spend:.3f}")

    res = pd.DataFrame(results)
    res.to_parquet(OUT / "probe.parquet", index=False)

    ans = res[res["llm_direction"].notna()].copy()
    ans["pred_up"] = (ans["llm_direction"] == "Rise").astype(int)
    ans["true_up"] = (ans["fwd_r5"] > 0).astype(int)

    def acc(pred, true):
        return float((pred == true).mean())

    ci = clustered_bootstrap(acc, ans["pred_up"].values, ans["true_up"].values,
                             ans["trade_date"].values, n_boot=500)
    summary = {
        "market": "A-shares (CN)", "model": MODEL,
        "n": int(len(res)), "n_parsed": int(len(ans)),
        "parse_fail_rate": float(res["llm_direction"].isna().mean()),
        "no_context_accuracy": acc(ans["pred_up"].values, ans["true_up"].values),
        "accuracy_clustered_ci95": [ci["lo"], ci["hi"]],
        "base_rate_up": float(ans["true_up"].mean()),
        "spend_usd": round(spend, 4),
        "contrast_finben_no_context": "ACL18 73.3% / BigData22 71.9% / CIKM18 80.3%",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
