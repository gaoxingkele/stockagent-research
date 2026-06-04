"""LO2 -- hermetic smoke for the long-only eval machinery."""
import numpy as np
import pandas as pd

from src.onset.run_long_only import eval_long_only


def test_eval_long_only_keys():
    rng = np.random.default_rng(0)
    n = 400
    ret = rng.normal(0, 0.03, n)
    df = pd.DataFrame({"trade_date": np.repeat(np.arange(n // 20), 20).astype(str),
                       "_fwd_r5": ret, "sig": 2.0 * ret + rng.normal(0, 0.03, n)})
    r = eval_long_only(df, k_frac=0.2)
    assert {"gross", "net"} <= set(r.keys())
    for arm in ("gross", "net"):
        assert {"mean_per_period", "annualized_sharpe", "mean_ci95"} <= set(r[arm].keys())
    assert r["net"]["mean_per_period"] < r["gross"]["mean_per_period"]   # cost lowers net
