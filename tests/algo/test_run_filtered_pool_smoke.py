"""FILT2 -- hermetic smoke for pool_excess."""
import numpy as np
import pandas as pd

from src.onset.run_filtered_pool import pool_excess


def test_pool_excess_runs():
    rng = np.random.default_rng(0)
    n = 300
    ret = rng.normal(0, 0.03, n)
    df = pd.DataFrame({"trade_date": np.repeat(np.arange(n // 30), 30).astype(str), "_fwd_r5": ret})
    # pool = the high-ret half -> positive excess
    in_pool = pd.Series(ret > np.median(ret), index=df.index)
    ex = pool_excess(df, in_pool)
    assert len(ex) > 0
    assert ex.mean() > 0
