"""COMBO1 -- hermetic smoke for combo_eval."""
import numpy as np
import pandas as pd

from src.onset.run_production_faithful import combo_eval


def test_combo_eval_keys():
    rng = np.random.default_rng(0)
    n = 150
    idx = [f"d{i}" for i in range(n)]
    mr = pd.Series(rng.normal(0.002, 0.02, n), index=idx)
    in_market = pd.Series(mr.values > mr.quantile(0.2), index=idx)
    sel = pd.Series(0.003, index=idx)
    d = combo_eval(mr, in_market, sel)
    assert {"buy_hold_sharpe", "timed_sharpe", "timed_plus_selected_sharpe"} <= set(d.keys())
