"""ROB1 -- hermetic smoke for robustness helpers."""
import numpy as np
import pandas as pd

from src.onset.run_robustness import topk_sensitivity, liquidity_subset


def _df(seed=0):
    rng = np.random.default_rng(seed); n = 400
    ret = rng.normal(0, 0.03, n)
    return pd.DataFrame({"trade_date": np.repeat(np.arange(n // 20), 20).astype(str),
                         "_fwd_r5": ret, "sig": 2.0 * ret + rng.normal(0, 0.03, n),
                         "amount": rng.uniform(1e6, 1e8, n)})


def test_topk_sensitivity_keys():
    s = topk_sensitivity(_df(), ks=(0.05, 0.1, 0.2))
    assert set(s.keys()) == {"k5", "k10", "k20"}
    assert all("annualized_sharpe" in v for v in s.values())


def test_liquidity_subset_filters():
    df = _df()
    liq = liquidity_subset(df, top_frac=0.5)
    assert len(liq) <= len(df)
    assert liq["amount"].min() >= df["amount"].median() - 1e-6
