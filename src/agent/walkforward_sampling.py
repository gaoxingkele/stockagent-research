"""Walk-forward random sampling for W1 — 3 splits × 2000 anchors.

Unlike `sampling.py` (stratified for PoC), this script does uniform random
sampling within each test window so that the onset rate reflects reality
(~8% expert pattern triggers, not 25%).

Output: data/processed/wf_samples_split{1,2,3}.parquet
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.onset.expert_pattern import bullish_onset_rules
from src.onset.disaster_filter import (
    compute_daily_market_signals,
    compute_disaster_signals,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
D1 = ROOT / "data/processed/ashares_tushare_v1.parquet"
OUT_DIR = ROOT / "data/processed"

# 3 walk-forward splits (test windows only here; train windows handled by W1.2)
SPLITS = {
    1: ("20250401", "20250701"),
    2: ("20250701", "20251001"),
    3: ("20251001", "20260101"),
}
N_PER_SPLIT = 2000
SEED = 42
HIST_DAYS = 30


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rng = np.random.default_rng(SEED)

    logger.info("Loading D1 + computing market signals (one-shot)...")
    df = pd.read_parquet(D1)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # Expert pattern + market signals (full panel one shot — reused per split)
    expert = bullish_onset_rules(df)
    for c in ["bottoms_rising", "above_5d_low_5pct", "ma_pattern_ok",
              "volume_boost", "is_bullish_onset", "onset_score"]:
        df[f"_exp_{c}"] = expert[c]

    m = compute_daily_market_signals(
        df[["ts_code", "trade_date", "pct_chg", "amount", "industry", "total_mv"]]
    )
    d = compute_disaster_signals(m)
    market = m.join(d[["signal_A_index", "signal_B_volume", "signal_C_sector",
                        "outer_vote_count", "is_disaster_month"]])

    # Forward 5-day return
    df["_fwd_r5"] = df.groupby("ts_code")["close"].shift(-5) / df["close"] - 1

    # Per split — random sample with valid fwd_r5
    for split_id, (start, end) in SPLITS.items():
        logger.info(f"=== Split {split_id}: test {start} -> {end} ===")
        pool = df[(df["trade_date"] >= start)
                   & (df["trade_date"] < end)
                   & df["_fwd_r5"].notna()].copy()
        logger.info(f"  Pool size: {len(pool):,}")
        if len(pool) < N_PER_SPLIT:
            logger.warning(f"  Pool < N_PER_SPLIT, sampling all {len(pool)}")
            sub = pool.copy()
        else:
            sub = pool.sample(N_PER_SPLIT, random_state=rng.integers(0, 10**9))
        sub = sub.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        logger.info(f"  Sampled {len(sub):,} anchors")
        logger.info(f"  Onset rate (exp_is_bullish_onset True): {sub['_exp_is_bullish_onset'].mean():.2%}")
        logger.info(f"  Mean fwd_r5: {sub['_fwd_r5'].mean()*100:+.2f}%")
        logger.info(f"  Std fwd_r5: {sub['_fwd_r5'].std()*100:.2f}%")

        # Attach 30-day history
        df_min = df[["ts_code", "trade_date", "open", "high", "low",
                      "close", "vol", "pct_chg"]]
        histories = []
        for _, row in sub.iterrows():
            ts, td = row["ts_code"], row["trade_date"]
            stock_hist = df_min[(df_min["ts_code"] == ts)
                                  & (df_min["trade_date"] <= td)].tail(HIST_DAYS)
            histories.append(stock_hist.to_dict(orient="records"))
        sub["_history"] = histories

        # Market context for each anchor
        mkt_cols = ["sh_index_pct", "gem_index_pct", "amount_ratio_5_20",
                    "limit_down_count", "up_stock_pct", "industry_red_pct",
                    "signal_A_index", "signal_B_volume", "is_disaster_month"]
        for c in mkt_cols:
            sub[f"_mkt_{c}"] = sub["trade_date"].map(market[c])

        # Keep columns
        PROMPT_FACTORS = [
            "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60",
            "rsi_14", "kdj_k", "kdj_d", "macd", "macd_hist",
            "boll_pct", "atr_pct", "vol_ratio_5", "vol_ratio_20",
            "winner_rate", "main_net", "holder_pct",
            "total_mv", "pe", "pb", "industry",
            "market_score_adj", "mf_strength", "mf_consecutive",
            "bias_10", "bias_20", "lr_slope_20", "channel_pos_60",
        ]
        keep_factor_cols = [c for c in PROMPT_FACTORS if c in sub.columns]
        keep = (
            ["ts_code", "trade_date",
             "open", "high", "low", "close", "vol", "pct_chg",
             "_history", "_fwd_r5"]
            + [f"_exp_{c}" for c in ["bottoms_rising", "above_5d_low_5pct",
                                       "ma_pattern_ok", "volume_boost",
                                       "is_bullish_onset", "onset_score"]]
            + [f"_mkt_{c}" for c in mkt_cols]
            + keep_factor_cols
        )
        out = sub[keep].copy()
        out["_history"] = out["_history"].apply(json.dumps)
        out["_split_id"] = split_id

        path = OUT_DIR / f"wf_samples_split{split_id}.parquet"
        out.to_parquet(path, index=False)
        logger.info(f"  Saved {len(out):,} -> {path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
