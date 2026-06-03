"""ID3 — hermetic smoke test for the A-share identification machinery."""
import numpy as np
import pandas as pd

from src.identify.run_ashare_identify import identify_from_scored


def test_identify_from_scored_runs_and_has_keys():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "lgbm_pump_ratio": rng.normal(size=n),
        "raw_p_up": rng.uniform(size=n),
        "expert_p_up": rng.uniform(size=n),
        "_fwd_r5": rng.normal(size=n),
        "trade_date": np.repeat(np.arange(n // 10), 10).astype(str),
    })
    validity = {"holds": True, "margin": -0.01, "reason": "synthetic"}
    res = identify_from_scored(df, "lgbm_pump_ratio", "_fwd_r5", "trade_date",
                               {"raw": "raw_p_up", "expert": "expert_p_up"}, validity)
    assert res["identified"] == "yes"
    for label in ("raw", "expert"):
        c = res["contribution"][label]
        assert {"mean", "lo", "hi"} <= set(c.keys())
        assert np.isfinite(c["mean"])
