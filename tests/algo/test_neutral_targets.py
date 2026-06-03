"""NB1 — neutral target builder. CPU, hermetic."""
import numpy as np
import pandas as pd

from src.identify.neutral_targets import market_neutral, sector_neutral, decompose


def _df(seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    return pd.DataFrame({
        "trade_date": np.repeat(np.arange(n // 20), 20).astype(str),
        "industry": rng.integers(0, 4, n).astype(str),
        "_fwd_r5": rng.normal(size=n),
    })


def test_market_neutral_sums_to_zero_per_date():
    df = _df()
    idio, syst = market_neutral(df)
    g = pd.DataFrame({"d": df["trade_date"], "i": idio}).groupby("d")["i"].sum()
    assert np.allclose(g.values, 0, atol=1e-9)
    assert np.allclose((idio + syst).values, df["_fwd_r5"].values)   # raw = idio + systematic


def test_sector_neutral_sums_to_zero_per_date_industry():
    df = _df()
    idio, _ = sector_neutral(df)
    g = pd.DataFrame({"d": df["trade_date"], "s": df["industry"], "i": idio}).groupby(["d", "s"])["i"].sum()
    assert np.allclose(g.values, 0, atol=1e-9)


def test_decompose_columns():
    out = decompose(_df())
    for c in ("mkt_neutral", "mkt_systematic", "sec_neutral", "sec_systematic"):
        assert c in out.columns
