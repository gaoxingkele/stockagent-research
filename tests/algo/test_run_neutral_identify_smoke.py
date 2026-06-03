"""NB3 — hermetic smoke for neutral identification machinery."""
import numpy as np
import pandas as pd

from src.identify.run_neutral_identify import neutral_identify, long_short


def _df(seed=0):
    rng = np.random.default_rng(seed)
    n = 400
    return pd.DataFrame({
        "trade_date": np.repeat(np.arange(n // 20), 20).astype(str),
        "industry": rng.integers(0, 4, n).astype(str),
        "_fwd_r5": rng.normal(size=n),
        "lgbm_pump_ratio": rng.normal(size=n),
        "raw_p_up": rng.uniform(size=n),
        "expert_p_up": rng.uniform(size=n),
    })


def test_long_short_keys():
    ls = long_short(_df(), "lgbm_pump_ratio")
    assert {"ls_mean_per_period", "annualized_sharpe", "n_dates"} <= set(ls.keys())


def test_neutral_identify_structure():
    res = neutral_identify(_df(), "lgbm_pump_ratio",
                           {"raw": "raw_p_up", "expert": "expert_p_up"},
                           {"holds": True})
    assert res["identified"] == "yes"
    assert "market_neutral" in res["selection"] and "raw" in res["selection"]["market_neutral"]
    assert {"mean", "lo", "hi"} <= set(res["selection"]["market_neutral"]["raw"].keys())
    assert "baseline" in res["long_short"]
