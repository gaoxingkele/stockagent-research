"""TRD1 -- hermetic smoke: diagnose() classifies directional vs sign-blind."""
import numpy as np
import pandas as pd

from src.onset.run_tradability import diagnose


def _df(kind, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, n)
    x = rng.standard_normal(n)
    if kind == "directional":
        sign = np.where(z == 0, -1.0, 1.0)
        y = sign * 0.5 * x + rng.standard_normal(n)
    else:                                    # sign-blind: variance only
        y = (0.3 + np.abs(x)) * rng.standard_normal(n)
    return pd.DataFrame({"feat": x, "_fwd_r5": y, "rg_trend": z,
                         "rg_vol": z, "rg_disaster": np.zeros(n, int)})


def test_directional_flagged_tradable():
    d = diagnose(_df("directional"), "feat", "trend", "_fwd_r5")
    assert d["looks_directional_tradable"] is True
    assert d["best_state_mono"] > 0.6


def test_sign_blind_flagged_not_tradable():
    d = diagnose(_df("sign_blind"), "feat", "trend", "_fwd_r5")
    assert d["looks_directional_tradable"] is False
    assert d["variance_vs_mean"]["directional_fraction"] < 0.5
