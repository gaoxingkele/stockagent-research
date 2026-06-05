"""MI3 -- hermetic smoke: per-year stability machinery + stable/unstable logic."""
import numpy as np
import pandas as pd

from src.onset.run_mi_stability import per_year


def _panel(stable: bool, seed=0):
    """Build a small panel with a trend regime that flips a feature->target
    relation. stable=True: the interaction is present EVERY year; stable=False:
    only in 2022."""
    rng = np.random.default_rng(seed)
    rows = []
    for yr in ("2022", "2023", "2024"):
        active = stable or (yr == "2022")
        for i in range(3000):
            z = int(rng.integers(0, 2))
            f = rng.standard_normal()
            if active:
                y = (1.0 if z == 1 else -1.0) * np.sign(f) + 0.1 * rng.standard_normal()
            else:
                y = rng.standard_normal()           # no relation this year
            rows.append({"trade_date": f"{yr}0101", "f": f, "_fwd_r5": y,
                         "rg_trend": z})
    return pd.DataFrame(rows)


def test_stable_interaction_significant_each_year():
    df = _panel(stable=True)
    yr = per_year(df, "f", "trend", "_fwd_r5", n_perm=120)
    assert len(yr) == 3
    n_pos_sig = sum(1 for v in yr.values()
                    if v["interaction"] > 0 and v["p_value"] < 0.05)
    assert n_pos_sig == 3


def test_unstable_interaction_only_one_year():
    df = _panel(stable=False)
    yr = per_year(df, "f", "trend", "_fwd_r5", n_perm=120)
    n_pos_sig = sum(1 for v in yr.values()
                    if v["interaction"] > 0 and v["p_value"] < 0.05)
    assert n_pos_sig <= 1
