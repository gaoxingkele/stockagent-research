"""E3 — score a FLARE stock-movement test split with a real LLM.

Sends each benchmark `query` (the exact prompt FinBen/PIXIU use) to the LLM and
parses a Rise/Fall direction + optional confidence. Writes
results/e3_<ds>/llm_test.parquet with [id, llm_direction, llm_conf, llm_p],
where the FLARE convention is gold=1 <-> "Fall", so llm_p = P(gold=1) = P(Fall),
directly comparable to the baseline's P(gold=1).

Concurrent (thread pool), resumable (skips ids already in llm_test.parquet),
checkpoints periodically, and tracks spend.

Usage:
    .venv-xpu\\Scripts\\python.exe -m src.eval_e3.run_llm --dataset acl --model gemini-3.5-flash
    .venv-xpu\\Scripts\\python.exe -m src.eval_e3.run_llm --dataset acl --limit 20   # smoke
"""
from __future__ import annotations
import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.agent.llm_client import call_llm

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/e3_flare"
RESULTS = ROOT / "results"

SYSTEM = (
    "You are a financial analyst predicting next-day stock price direction from "
    "recent price history and tweets. Answer with EXACTLY one word: Rise or Fall."
)
_DIR = re.compile(r"\b(rise|fall)\b", re.I)
_CONF = re.compile(r"confidence\D{0,5}(\d{1,3})", re.I)

_lock = threading.Lock()


def parse(content: str) -> tuple[str | None, float]:
    d = _DIR.search(content or "")
    direction = d.group(1).capitalize() if d else None
    c = _CONF.search(content or "")
    conf = float(c.group(1)) if c else 60.0
    conf = max(0.0, min(100.0, conf))
    return direction, conf


def to_p(direction: str | None, conf: float) -> float:
    """P(gold=1)=P(Fall). Unknown -> 0.5 (no signal)."""
    if direction is None:
        return 0.5
    mag = 0.5 * (conf / 100.0)
    return (0.5 + mag) if direction == "Fall" else (0.5 - mag)


def build_user(row, no_context: bool) -> str:
    if not no_context:
        return row["query"]
    # ablation: ticker + date only, NO price history or tweets -> isolates
    # whether the model is recalling memorized history vs analysing context.
    return (f"Will the closing price of ${row['ticker']} go upwards or downwards "
            f"at {row['date']}? Please indicate either Rise or Fall.")


def score_one(row, model, no_context=False, max_tokens=256):
    try:
        r = call_llm(system=SYSTEM, user=build_user(row, no_context), model=model,
                     temperature=0.0, max_tokens=max_tokens)
        direction, conf = parse(r.content)
        return {"id": row["id"], "llm_direction": direction, "llm_conf": conf,
                "llm_p": to_p(direction, conf), "cost_usd": r.cost_usd}
    except Exception as e:
        return {"id": row["id"], "llm_direction": None, "llm_conf": 60.0,
                "llm_p": 0.5, "cost_usd": 0.0, "error": str(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="acl")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None, help="random subsample size")
    ap.add_argument("--no-context", action="store_true", help="ticker+date only ablation")
    ap.add_argument("--tag", default=None, help="output suffix (separate file, no resume)")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="raise to avoid truncating models that emit hidden reasoning")
    args = ap.parse_args()

    te = pd.read_parquet(BASE / args.dataset / "parsed_test.parquet")
    if args.sample:
        te = te.sample(args.sample, random_state=42)
    if args.limit:
        te = te.head(args.limit)
    out_dir = RESULTS / f"e3_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"llm_test_{args.tag}.parquet" if args.tag else "llm_test.parquet")

    done = set()
    results = []
    if out_path.exists() and not args.limit and not args.sample and not args.tag:
        prev = pd.read_parquet(out_path)
        results = prev.to_dict("records")
        done = set(prev["id"])
        print(f"Resuming: {len(done)} already scored")

    todo = te[~te["id"].isin(done)]
    print(f"Scoring {len(todo)} anchors with {args.model} ({args.workers} workers)")

    spend = 0.0
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_one, row, args.model, args.no_context, args.max_tokens): row["id"]
                for _, row in todo.iterrows()}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            spend += res.get("cost_usd", 0.0)
            n_done += 1
            if n_done % 200 == 0:
                with _lock:
                    pd.DataFrame(results).to_parquet(out_path, index=False)
                print(f"  {n_done}/{len(todo)} done | spend ${spend:.3f}")

    df = pd.DataFrame(results)
    df.to_parquet(out_path, index=False)
    n_err = df["llm_direction"].isna().sum()
    print(f"\nDone: {len(df)} rows | parse-fail/err {n_err} | total spend ${spend:.3f}")
    print(f"direction dist: {df['llm_direction'].value_counts(dropna=False).to_dict()}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
