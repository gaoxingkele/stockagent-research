"""K6 -- synthesis + explicit alpha1 verdict (SIGN-K1).

Reads results/candle/{lgbm,seq,ablation}.json and applies the SIGN-K1 rule:
a tradable edge (alpha1) is REAL only if the pooled, NET-of-cost market-neutral
long-short mean CI excludes 0 AND the net Sharpe is positive in >= 2/3 splits.
Anything that only works pooled-without-per-split, or on a single split, is
labelled PROMISING-UNCONFIRMED or single-split/overfit, not alpha.

Run: .venv-xpu\\Scripts\\python.exe -m src.onset.summarize_candle
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAND = ROOT / "results/candle"
FIG_DIR = ROOT / "paper/sections/figures"


def alpha_verdict(pooled_net_ci, per_split_net_sharpes) -> str:
    """SIGN-K1 verdict. pooled_net_ci = [lo,hi] of pooled net long-short mean;
    per_split_net_sharpes = list of per-split net Sharpe (or None if unavailable)."""
    pooled_sig = pooled_net_ci is not None and (pooled_net_ci[0] > 0 or pooled_net_ci[1] < 0)
    if per_split_net_sharpes is None:
        return "PROMISING (pooled net CI excludes 0, per-split unconfirmed)" if pooled_sig else "null"
    n_pos = sum(1 for s in per_split_net_sharpes if s is not None and s > 0)
    n = len(per_split_net_sharpes)
    if pooled_sig and n_pos >= max(2, (2 * n + 2) // 3):
        return "REAL (alpha1): pooled net CI excludes 0 AND net>0 in >=2/3 splits"
    if pooled_sig:
        return f"PROMISING: pooled net CI excludes 0 but net>0 only {n_pos}/{n} splits"
    if n_pos == 1:
        return "single-split / overfit"
    return "null / not cost-surviving"


def _per_split_net(obj):
    ps = obj.get("per_split", {})
    return [ps[k]["long_short"].get("net_sharpe") for k in sorted(ps)] if ps else None


def run_real() -> dict:
    lgbm = json.loads((CAND / "lgbm.json").read_text(encoding="utf-8"))
    seq = json.loads((CAND / "seq.json").read_text(encoding="utf-8"))
    abl = json.loads((CAND / "ablation.json").read_text(encoding="utf-8"))

    out = {"verdicts": {}, "pooled_rank_ic": {}, "pooled_net_sharpe": {}}
    out["verdicts"]["candle_flat_lgbm (K3)"] = alpha_verdict(
        lgbm["pooled"]["long_short"].get("net_mean_ci95"), _per_split_net(lgbm))
    out["verdicts"]["candle_seq (K4)"] = alpha_verdict(
        seq["pooled"]["long_short"].get("net_mean_ci95"), _per_split_net(seq))
    def _abl_persplit(name):
        ps = abl["by_set"][name].get("per_split_net_sharpe")
        return list(ps.values()) if ps else None
    out["verdicts"]["factors_plus_candle (K5)"] = alpha_verdict(
        abl["by_set"]["factors_plus_candle"].get("net_mean_ci95"), _abl_persplit("factors_plus_candle"))
    out["verdicts"]["factors_only (K5)"] = alpha_verdict(
        abl["by_set"]["factors"].get("net_mean_ci95"), _abl_persplit("factors"))

    out["pooled_rank_ic"] = {
        "candle_flat": lgbm["pooled"]["rank_ic_market_neutral"]["mean"],
        "candle_seq": seq["pooled"]["rank_ic_market_neutral"]["mean"],
        "factors": abl["by_set"]["factors"]["rank_ic_market_neutral"]["mean"],
        "factors_plus_candle": abl["by_set"]["factors_plus_candle"]["rank_ic_market_neutral"]["mean"]}
    out["pooled_net_sharpe"] = {
        "candle_flat": lgbm["pooled"]["long_short"]["net_sharpe"],
        "candle_seq": seq["pooled"]["long_short"]["net_sharpe"],
        "factors": abl["by_set"]["factors"]["net_sharpe"],
        "factors_plus_candle": abl["by_set"]["factors_plus_candle"]["net_sharpe"]}
    out["incremental_net_sharpe_from_candle"] = abl["summary"].get("incremental_net_sharpe_from_candle")

    CAND.mkdir(parents=True, exist_ok=True)
    md = ["# Candlestick-onset alpha1 verdict\n", "## Pooled market-neutral RankIC / net long-short Sharpe\n",
          "| model | RankIC | net Sharpe |", "|---|---|---|"]
    for k in out["pooled_rank_ic"]:
        md.append(f"| {k} | {out['pooled_rank_ic'][k]:+.3f} | {out['pooled_net_sharpe'][k]:+.2f} |")
    md += [f"\nIncremental net Sharpe from adding candle geometry: **{out['incremental_net_sharpe_from_candle']:+.2f}**\n",
           "## SIGN-K1 verdicts\n", "| model | verdict |", "|---|---|"]
    for k, v in out["verdicts"].items():
        md.append(f"| {k} | {v} |")
    (CAND / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _plot(out)
    return out


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    ks = list(out["pooled_rank_ic"].keys())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(ks, [out["pooled_rank_ic"][k] for k in ks], color="#1f77b4")
    a1.axhline(0, color="k", lw=0.8); a1.set_title("pooled market-neutral RankIC", fontsize=10)
    a1.tick_params(axis="x", rotation=30, labelsize=7)
    a2.bar(ks, [out["pooled_net_sharpe"][k] for k in ks], color="#d62728")
    a2.axhline(0, color="k", lw=0.8); a2.set_title("pooled NET long-short Sharpe (after 0.4% cost)", fontsize=10)
    a2.tick_params(axis="x", rotation=30, labelsize=7)
    fig.tight_layout(); FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "candle_alpha_summary.png", dpi=150)


def main():
    print(json.dumps(run_real(), indent=2))


if __name__ == "__main__":
    main()
