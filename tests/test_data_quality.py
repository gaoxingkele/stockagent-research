"""Data quality tests (execution_plan §1.6).

These run against the actual D1 parquet once built — skipped if data not present.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest

D1_PATH = Path("data/processed/ashares_tushare_v1.parquet")


@pytest.fixture(scope="module")
def d1():
    if not D1_PATH.exists():
        pytest.skip(f"D1 not built yet at {D1_PATH}. Run `make data_d1`.")
    return pd.read_parquet(D1_PATH)


def test_d1_has_required_columns(d1):
    required = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    for col in required:
        assert col in d1.columns, f"Missing column: {col}"


def test_d1_no_duplicate_keys(d1):
    dup = d1.duplicated(subset=["ts_code", "trade_date"])
    assert not dup.any(), f"{dup.sum()} duplicate (ts_code, trade_date) rows"


def test_d1_date_format(d1):
    assert d1["trade_date"].astype(str).str.match(r"^\d{8}$").all()


def test_d1_st_excluded(d1):
    """ST stocks must be filtered at source (feedback_st_exclude_at_source)."""
    if "name" in d1.columns:
        st_rows = d1[d1["name"].str.contains("ST", case=False, na=False)]
        assert len(st_rows) == 0, f"{len(st_rows)} ST rows leaked into D1"


def test_d1_survivorship(d1):
    """D1 should include historical delisted stocks (no survivorship bias)."""
    n_stocks = d1["ts_code"].nunique()
    assert n_stocks > 4000, f"Only {n_stocks} stocks — likely missing delisted ones"


def test_d1_price_no_extreme_jumps(d1):
    """Adjacent-day price ratio should not exceed 30% unless corporate action."""
    d1_sorted = d1.sort_values(["ts_code", "trade_date"])
    d1_sorted["prev_close"] = d1_sorted.groupby("ts_code")["close"].shift(1)
    d1_sorted["jump"] = (d1_sorted["close"] / d1_sorted["prev_close"] - 1).abs()
    extreme = d1_sorted["jump"] > 0.30
    # A 股有 10% / 20% 涨跌停, jump > 30% 应非常少 (corporate action only)
    rate = extreme.mean()
    assert rate < 0.001, f"{rate:.4%} rows with >30% jump"
