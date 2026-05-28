"""Build D1 dataset: A-shares Tushare panel data.

Reuses production assets from `D:/aicoding/stockagent-analysis/`:
  - output/tushare_cache/daily/*.parquet   (OHLCV)
  - output/tushare_cache/stock_basic.parquet
  - output/factor_lab_3y/                   (153 factors)

Output:
  data/processed/ashares_tushare_v1.parquet  (full panel)
  data/processed/symbol_universe.yaml
  data/processed/trading_calendar.csv
"""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROD_ROOT = Path("D:/aicoding/stockagent-analysis")
TUSHARE_DAILY = PROD_ROOT / "output/tushare_cache/daily"
STOCK_BASIC = PROD_ROOT / "output/tushare_cache/stock_basic.parquet"
FACTOR_LAB_MAIN = PROD_ROOT / "output/factor_lab_3y/factor_groups"
FACTOR_LAB_EXT = PROD_ROOT / "output/factor_lab_3y/factor_groups_extension"

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = RESEARCH_ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ohlcv() -> pd.DataFrame:
    """Concat all daily parquets from production tushare_cache."""
    files = sorted(TUSHARE_DAILY.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No daily files in {TUSHARE_DAILY}")
    logger.info(f"Loading {len(files)} daily files")
    parts = [pd.read_parquet(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def load_factors() -> pd.DataFrame:
    """Load 153 factors from production factor_lab.

    Strategy: concat factor_groups/ (main 3-year history) + factor_groups_extension/ (recent increments).
    Dedup by (ts_code, trade_date), keep last occurrence (extension is more recent).
    """
    parts = []
    if FACTOR_LAB_MAIN.exists():
        main_files = sorted(FACTOR_LAB_MAIN.glob("group_*.parquet"))
        logger.info(f"Loading {len(main_files)} main factor groups from {FACTOR_LAB_MAIN.name}")
        for f in main_files:
            parts.append(pd.read_parquet(f))
    if FACTOR_LAB_EXT.exists():
        ext_files = sorted(FACTOR_LAB_EXT.glob("*_ext*.parquet"))
        logger.info(f"Loading {len(ext_files)} extension factor files from {FACTOR_LAB_EXT.name}")
        for f in ext_files:
            parts.append(pd.read_parquet(f))
    if not parts:
        logger.warning("No factor files found, returning empty")
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    n_before = len(df)
    df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last").reset_index(drop=True)
    logger.info(f"Factor dedup: {n_before:,} -> {len(df):,} rows")
    return df


def filter_st_at_source(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude ST stocks from training set (feedback_st_exclude_at_source)."""
    basic = pd.read_parquet(STOCK_BASIC)
    st_mask = basic["name"].str.contains("ST", case=False, na=False) if "name" in basic.columns else pd.Series([False] * len(basic))
    st_codes = set(basic.loc[st_mask, "ts_code"])
    n_before = df["ts_code"].nunique()
    df = df[~df["ts_code"].isin(st_codes)].copy()
    n_after = df["ts_code"].nunique()
    logger.info(f"ST filter: {n_before} -> {n_after} stocks ({len(st_codes)} ST excluded)")
    return df


def build_universe(df: pd.DataFrame) -> dict:
    """Build symbol universe with first/last trade date per stock."""
    grp = df.groupby("ts_code")["trade_date"].agg(["min", "max", "count"])
    universe = {
        "total_stocks": len(grp),
        "date_range": [df["trade_date"].min(), df["trade_date"].max()],
        "stocks": grp.reset_index().to_dict(orient="records"),
    }
    return universe


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("=== Building D1: A-shares Tushare ===")

    ohlcv = load_ohlcv()
    logger.info(f"OHLCV: {len(ohlcv):,} rows, {ohlcv['ts_code'].nunique()} stocks")

    ohlcv = filter_st_at_source(ohlcv)

    factors = load_factors()
    logger.info(f"Factors: {len(factors):,} rows" if len(factors) else "Factors: (empty, will join later)")

    # 合并 factor (如果有)
    if len(factors):
        panel = ohlcv.merge(factors, on=["ts_code", "trade_date"], how="left")
    else:
        panel = ohlcv

    # 落盘
    out_panel = OUT_DIR / "ashares_tushare_v1.parquet"
    panel.to_parquet(out_panel, index=False)
    logger.info(f"Panel saved: {out_panel} ({len(panel):,} rows)")

    # universe
    universe = build_universe(panel)
    out_univ = OUT_DIR / "symbol_universe.yaml"
    with open(out_univ, "w", encoding="utf-8") as f:
        yaml.safe_dump({"total_stocks": universe["total_stocks"],
                        "date_range": universe["date_range"]}, f)
    logger.info(f"Universe: {out_univ} ({universe['total_stocks']} stocks)")

    # 交易日历
    cal = sorted(panel["trade_date"].unique())
    pd.DataFrame({"trade_date": cal}).to_csv(OUT_DIR / "trading_calendar.csv", index=False)
    logger.info(f"Calendar: {len(cal)} days")

    logger.info("=== D1 build complete ===")


if __name__ == "__main__":
    main()
