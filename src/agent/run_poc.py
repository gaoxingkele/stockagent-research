"""Run LLM agent PoC: raw LLM vs V12.31-augmented LLM vs LGBM baseline.

Usage:
    python -m src.agent.run_poc --n 100 --model gemini-3.5-flash --tag smoke
    python -m src.agent.run_poc --n 1000 --model gemini-3.5-flash --tag full

Conditions:
  - raw:    minimal system prompt (no V12.31 expert knowledge)
  - expert: V12.31 expert system prompt (Round 1-3 captured knowledge)
  - LGBM:   model.predict on the same anchors (loaded from results/e1_1_fh_h5_v2)

Output:
  - results/poc_{tag}/predictions.parquet
  - results/poc_{tag}/metrics.json
  - results/poc_{tag}/summary.md
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.agent.llm_client import call_llm, _estimate_cost
from src.agent.prompt_builder import (
    load_expert_system_prompt, build_raw_system_prompt,
    build_user_prompt, parse_llm_response,
    load_expert_knowledge_body,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "data/processed/llm_poc_samples_v1.parquet"


def llm_predict_one(anchor: pd.Series, system_prompt: str, model: str,
                     expert_prefix: str = "") -> dict:
    """Predict P(up/neutral/down) for one anchor using the LLM."""
    user_p = build_user_prompt(anchor, expert_prefix=expert_prefix)
    try:
        r = call_llm(system=system_prompt, user=user_p, model=model,
                     temperature=0.1, max_tokens=200, max_retries=2)
        parsed = parse_llm_response(r.content)
        if parsed is None:
            return {"p_up": None, "p_neutral": None, "p_down": None,
                    "rationale": "PARSE_ERROR: " + r.content[:200],
                    "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                    "cost_usd": r.cost_usd, "raw_content": r.content}
        return {
            "p_up": float(parsed.get("p_up", 0)),
            "p_neutral": float(parsed.get("p_neutral", 0)),
            "p_down": float(parsed.get("p_down", 0)),
            "rationale": parsed.get("rationale", ""),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd": r.cost_usd,
            "raw_content": r.content,
        }
    except Exception as e:
        logger.error(f"LLM call failed for {anchor['ts_code']}@{anchor['trade_date']}: {e}")
        return {"p_up": None, "p_neutral": None, "p_down": None,
                "rationale": f"ERROR: {e}",
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0, "raw_content": ""}


def evaluate(df: pd.DataFrame, signal_col: str, target_col: str = "_fwd_r5") -> dict:
    """Compute RankIC + TopK return metrics on a sample DataFrame."""
    sub = df[[signal_col, target_col, "trade_date"]].dropna()
    if len(sub) < 10:
        return {"n_valid": len(sub), "rank_ic": np.nan, "rank_ic_p": np.nan,
                "top10_pct_mean_ret": np.nan, "top20_pct_mean_ret": np.nan}
    rho, p_value = spearmanr(sub[signal_col], sub[target_col])
    # Top 10% / Top 20% by signal — overall (not per-date due to small n in smoke)
    sorted_sub = sub.sort_values(signal_col, ascending=False)
    top10 = sorted_sub.head(max(1, len(sub) // 10))
    top20 = sorted_sub.head(max(1, len(sub) // 5))
    return {
        "n_valid": int(len(sub)),
        "rank_ic": float(rho),
        "rank_ic_p": float(p_value),
        "top10_pct_mean_ret": float(top10[target_col].mean()),
        "top10_pct_winrate": float((top10[target_col] > 0).mean()),
        "top20_pct_mean_ret": float(top20[target_col].mean()),
        "top20_pct_winrate": float((top20[target_col] > 0).mean()),
        "mean_target": float(sub[target_col].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="Number of anchors to process")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--tag", default="smoke", help="Run tag (output goes to results/poc_<tag>)")
    ap.add_argument("--conditions", nargs="+", default=["raw", "expert"],
                    choices=["raw", "expert"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=5,
                    help="ThreadPoolExecutor concurrent workers")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = ROOT / f"results/poc_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading samples from {SAMPLES}")
    samples = pd.read_parquet(SAMPLES)

    # Stratified subsample to n
    if args.n < len(samples):
        per_stratum = max(1, args.n // 3)
        rng = np.random.default_rng(args.seed)
        sub_parts = []
        for s, target_n in [("high", args.n * 25 // 100),
                             ("edge", args.n * 25 // 100),
                             ("low", args.n - 2 * (args.n * 25 // 100))]:
            pool = samples[samples["stratum"] == s]
            take = min(target_n, len(pool))
            sub_parts.append(pool.sample(take, random_state=rng.integers(0, 10**6)))
        sub = pd.concat(sub_parts, ignore_index=True)
    else:
        sub = samples
    logger.info(f"Working set: {len(sub)} anchors  (strata: {sub['stratum'].value_counts().to_dict()})")

    # Both conditions share the same short system prompt.
    # Expert knowledge for the "expert" condition is injected into the USER prompt
    # (head) to bypass any API-gateway system-prompt truncation.
    sys_prompt = load_expert_system_prompt()  # same as build_raw_system_prompt now
    expert_body = load_expert_knowledge_body()
    expert_prefix_map = {"raw": "", "expert": expert_body}

    # Run each condition
    all_results = sub[["ts_code", "trade_date", "stratum", "_fwd_r5",
                        "_exp_is_bullish_onset", "_exp_onset_score"]].copy()
    total_cost = 0.0
    t_start = time.time()
    for cond in args.conditions:
        logger.info(f"=== Running condition: {cond} (n={len(sub)}, model={args.model}, workers={args.workers}) ===")
        results = [None] * len(sub)  # preserve order
        t_cond = time.time()
        anchors_list = list(sub.iterrows())
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(llm_predict_one, anchor, sys_prompt, args.model,
                              expert_prefix_map[cond]): idx
                for idx, (_, anchor) in enumerate(anchors_list)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
                completed += 1
                if completed % 10 == 0 or completed == len(sub):
                    elapsed = time.time() - t_cond
                    rate = completed / elapsed
                    eta_min = (len(sub) - completed) / max(rate, 1e-6) / 60
                    ok = sum(1 for x in results if x and x["p_up"] is not None)
                    cost_so_far = sum(x["cost_usd"] for x in results if x)
                    logger.info(f"  {completed}/{len(sub)}  ok={ok}  rate={rate:.2f}/s  ETA={eta_min:.1f}m  cost=${cost_so_far:.4f}")
        rdf = pd.DataFrame(results)
        for c in ["p_up", "p_neutral", "p_down", "rationale",
                  "input_tokens", "output_tokens", "cost_usd"]:
            all_results[f"{cond}_{c}"] = rdf[c].values
        # Ratio signal (paper §2.7 C3)
        all_results[f"{cond}_pump_ratio"] = (
            rdf["p_up"].fillna(0) / (rdf["p_down"].fillna(0) + 0.01)
        ).values
        cond_cost = rdf["cost_usd"].sum()
        total_cost += cond_cost
        logger.info(f"=== {cond} done in {(time.time()-t_cond)/60:.1f}m  cost=${cond_cost:.4f} ===")

    elapsed_total = time.time() - t_start
    logger.info(f"All conditions done in {elapsed_total/60:.1f}m. Total cost: ${total_cost:.4f}")

    # Evaluate
    metrics = {"n": len(sub), "model": args.model, "tag": args.tag,
               "total_cost_usd": total_cost,
               "elapsed_minutes": elapsed_total / 60,
               "conditions": {}}

    # Stratified raw return reference
    metrics["stratum_mean_fwd_r5"] = sub.groupby("stratum")["_fwd_r5"].mean().to_dict()

    for cond in args.conditions:
        for sig_name, sig_col in [
            ("p_up", f"{cond}_p_up"),
            ("p_up_minus_p_down", None),  # computed below
            ("pump_ratio", f"{cond}_pump_ratio"),
        ]:
            if sig_name == "p_up_minus_p_down":
                all_results[f"{cond}_p_up_minus_p_down"] = (
                    all_results[f"{cond}_p_up"].fillna(0)
                    - all_results[f"{cond}_p_down"].fillna(0)
                )
                sig_col = f"{cond}_p_up_minus_p_down"
            m = evaluate(all_results, sig_col)
            metrics["conditions"].setdefault(cond, {})[sig_name] = m

    # Expert is_bullish_onset baseline (binary signal)
    m_exp_baseline = evaluate(all_results.assign(
        _exp_score=all_results["_exp_is_bullish_onset"].astype(int)
    ), "_exp_score")
    metrics["expert_pattern_baseline"] = m_exp_baseline

    # Save
    all_results.to_parquet(out_dir / "predictions.parquet", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    # Summary md
    lines = ["# PoC Run Summary\n",
             f"- Tag: {args.tag}", f"- Model: {args.model}", f"- N: {len(sub)}",
             f"- Conditions: {args.conditions}",
             f"- Elapsed: {elapsed_total/60:.1f} min",
             f"- Total cost: ${total_cost:.4f}",
             "\n## Stratum mean fwd_r5\n"]
    for s, v in metrics["stratum_mean_fwd_r5"].items():
        lines.append(f"  - {s}: {v*100:+.2f}%")
    lines.append("\n## Metrics by condition × signal\n")
    lines.append("| Condition | Signal | n | RankIC | Top10% ret | Top10% winrate | Top20% ret |")
    lines.append("|---|---|---|---|---|---|---|")
    for cond, sigs in metrics["conditions"].items():
        for sig, m in sigs.items():
            lines.append(
                f"| {cond} | {sig} | {m.get('n_valid', '-')} | "
                f"{m.get('rank_ic', float('nan')):.4f} | "
                f"{m.get('top10_pct_mean_ret', float('nan'))*100:+.2f}% | "
                f"{m.get('top10_pct_winrate', float('nan')):.2%} | "
                f"{m.get('top20_pct_mean_ret', float('nan'))*100:+.2f}% |"
            )
    lines.append("\n## Expert pattern baseline (binary is_bullish_onset)\n")
    m = metrics["expert_pattern_baseline"]
    lines.append(f"- RankIC: {m.get('rank_ic', float('nan')):.4f}")
    lines.append(f"- Top 10% mean ret: {m.get('top10_pct_mean_ret', float('nan'))*100:+.2f}%")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
