"""TIM2 -- hermetic smoke for the timing eval machinery."""
import numpy as np
import pandas as pd

from src.onset.run_timing import timing_eval


def test_timing_eval_keys_and_improvement():
    rng = np.random.default_rng(0)
    n = 200
    idx = [f"d{i}" for i in range(n)]
    mr = pd.Series(rng.normal(0.002, 0.02, n), index=idx)
    is_disaster = pd.Series(mr.values < mr.quantile(0.2), index=idx)   # worst 20% are disaster
    d = timing_eval(mr, is_disaster)
    for k in ("buy_hold_sharpe", "timed_sharpe", "incremental_timing", "disaster_frac", "n_switches"):
        assert k in d
    assert d["timed_sharpe"] > d["buy_hold_sharpe"]      # avoiding the worst periods helps
