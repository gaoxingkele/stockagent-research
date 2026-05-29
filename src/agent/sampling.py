"""Sample 1000 anchor points from D1 test split for LLM agent PoC.

Stratification:
  - 250 high-confidence onset (expert is_bullish_onset == True, onset_score == 4)
  - 250 edge cases       (onset_score == 3, partial trigger)
  - 500 non-onset        (onset_score <= 1)

Plus per-anchor context:
  - 165 model features (zero-filled NaN)
  - 30 days of historical OHLCV (for LLM to inspect price trajectory)
  - expert_pattern signals (bottoms_rising / above_5d_low_5pct / ma_pattern_ok / volume_boost)
  - daily market disaster signals at that date
  - ground truth: fwd_r5 (next-5-day return), used for evaluation only
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.onset.expert_pattern import bullish_onset_rules
from src.onset.disaster_filter import compute_daily_market_signals, compute_disaster_signals

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
D1 = ROOT / "data/processed/ashares_tushare_v1.parquet"
OUT = ROOT / "data/processed/llm_poc_samples_v1.parquet"

# Test split window (matches train.py default)
VAL_END = "20250701"
TEST_END = "20260601"

# Sample sizes per stratum
N_HIGH = 250   # is_bullish_onset True (score==4 preferred, score==3 OK)
N_EDGE = 250   # onset_score == 3 but is_bullish_onset == False
N_LOW = 500    # onset_score <= 1

HIST_DAYS = 30  # how many historical days to attach per anchor

# Subset of "important" factors to include in prompt (avoid context bloat)
PROMPT_FACTORS = [
    "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60",
    "rsi_14", "kdj_k", "kdj_d", "macd", "macd_hist",
    "boll_pct", "atr_pct", "vol_ratio_5", "vol_ratio_20",
    "winner_rate", "main_net", "holder_pct",
    "total_mv", "pe", "pb",
    "industry",
    # market context
    "market_score_adj", "mf_strength", "mf_consecutive",
    # additional momentum
    "bias_10", "bias_20", "lr_slope_20", "channel_pos_60",
]


def main():
    rng = np.random.default_rng(42)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info(f"Loading D1...")
    df = pd.read_parquet(D1)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    logger.info(f"D1: {len(df):,} rows × {df['ts_code'].nunique():,} stocks × {df['trade_date'].nunique():,} days")

    # Compute expert pattern over the full panel (we need history before sampling)
    logger.info("Computing expert_pattern (bullish_onset_rules)...")
    expert = bullish_onset_rules(df)
    for c in ["bottoms_rising", "above_5d_low_5pct", "ma_pattern_ok", "volume_boost",
              "is_bullish_onset", "onset_score"]:
        df[f"_exp_{c}"] = expert[c]

    # Daily market signals + disaster flags
    logger.info("Computing market signals...")
    m = compute_daily_market_signals(df[["ts_code", "trade_date", "pct_chg", "amount", "industry", "total_mv"]])
    d = compute_disaster_signals(m)
    market_daily = m.join(d[["signal_A_index", "signal_B_volume", "signal_C_sector",
                              "outer_vote_count", "is_disaster_month"]])
    market_daily.index.name = "trade_date"

    # Ground truth: fwd_r5
    logger.info("Computing fwd_r5...")
    df["_fwd_r5"] = df.groupby("ts_code")["close"].shift(-5) / df["close"] - 1

    # Restrict to test window with valid fwd_r5
    test = df[(df["trade_date"] >= VAL_END) & (df["trade_date"] < TEST_END) & df["_fwd_r5"].notna()].copy()
    logger.info(f"Test pool: {len(test):,} rows after fwd_r5 filter")

    # Strata
    high_pool = test[test["_exp_is_bullish_onset"] == True]
    edge_pool = test[(test["_exp_onset_score"] == 3) & (test["_exp_is_bullish_onset"] == False)]
    low_pool = test[test["_exp_onset_score"] <= 1]
    logger.info(f"Strata sizes: high={len(high_pool):,}, edge={len(edge_pool):,}, low={len(low_pool):,}")

    def sample_stratum(pool, n):
        n = min(n, len(pool))
        return pool.sample(n, random_state=rng.integers(0, 10**6))

    high = sample_stratum(high_pool, N_HIGH).assign(stratum="high")
    edge = sample_stratum(edge_pool, N_EDGE).assign(stratum="edge")
    low = sample_stratum(low_pool, N_LOW).assign(stratum="low")
    anchors = pd.concat([high, edge, low], ignore_index=False).reset_index(drop=True)
    logger.info(f"Sampled anchors: {len(anchors):,}")
    logger.info(f"  Mean fwd_r5 by stratum:")
    for s in ["high", "edge", "low"]:
        sub = anchors[anchors["stratum"] == s]
        logger.info(f"    {s:5s}: n={len(sub):>4d}  mean fwd_r5={sub['_fwd_r5'].mean()*100:+.2f}%  std={sub['_fwd_r5'].std()*100:.2f}%")

    # Attach historical OHLCV for each anchor
    logger.info(f"Attaching {HIST_DAYS}-day history per anchor...")
    df_min = df[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "pct_chg"]]
    histories = []
    for _, row in anchors.iterrows():
        ts = row["ts_code"]
        td = row["trade_date"]
        stock_hist = df_min[(df_min["ts_code"] == ts) & (df_min["trade_date"] <= td)].tail(HIST_DAYS)
        histories.append(stock_hist.to_dict(orient="records"))
    anchors["_history"] = histories

    # Attach market context for each anchor's date
    logger.info("Attaching market context per anchor's date...")
    mkt_cols = ["sh_index_pct", "gem_index_pct", "amount_ratio_5_20",
                "limit_down_count", "up_stock_pct", "industry_red_pct",
                "signal_A_index", "signal_B_volume", "is_disaster_month"]
    for c in mkt_cols:
        anchors[f"_mkt_{c}"] = anchors["trade_date"].map(market_daily[c])

    # Keep only the columns we need for the PoC
    keep_factor_cols = [c for c in PROMPT_FACTORS if c in anchors.columns]
    keep = (
        ["ts_code", "trade_date", "stratum",
         "open", "high", "low", "close", "vol", "pct_chg",
         "_history", "_fwd_r5"]
        + [f"_exp_{c}" for c in ["bottoms_rising", "above_5d_low_5pct",
                                  "ma_pattern_ok", "volume_boost",
                                  "is_bullish_onset", "onset_score"]]
        + [f"_mkt_{c}" for c in mkt_cols]
        + keep_factor_cols
    )
    out = anchors[keep].copy()

    logger.info(f"Final columns: {len(out.columns)}")
    logger.info(f"Saving to {OUT}...")
    # pyarrow doesn't like list-of-dict columns; serialize history as JSON string for portability
    import json
    out["_history"] = out["_history"].apply(json.dumps)
    out.to_parquet(OUT, index=False)
    logger.info(f"Saved {len(out):,} samples to {OUT}")


if __name__ == "__main__":
    main()
