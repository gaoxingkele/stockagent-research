"""MI2 -- hermetic smoke: the probe machinery runs and the verdict logic works."""
import numpy as np
import pandas as pd

from src.onset.run_mi_probe import probe, _verdict


def _synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, n)
    f_cond = rng.standard_normal(n)
    # target depends on f_cond ONLY given z (sign flips by regime) -> conditional
    sign = np.where(z == 0, -1.0, 1.0)
    y = sign * np.sign(f_cond) + 0.1 * rng.standard_normal(n)
    f_noise = rng.standard_normal(n)
    return pd.DataFrame({"f_cond": f_cond, "f_noise": f_noise, "_fwd_r5": y,
                         "rg_trend": z, "rg_vol": rng.integers(0, 3, n),
                         "rg_disaster": np.zeros(n, dtype=int)})


def test_probe_detects_regime_added_info_and_verdict():
    df = _synthetic()
    res = probe(df, ["f_cond", "f_noise"], ["_fwd_r5"], n_perm=120)
    v = _verdict(res)
    # f_cond's relation to y FLIPS by regime -> the regime ADDS information
    assert v["n_hits"] >= 1
    assert v["verdict"].startswith("REGIME-ADDS-INFORMATION")
    hit_feats = {h["feature"] for h in v["hits"]}
    assert "f_cond" in hit_feats


def test_verdict_null_when_regime_irrelevant():
    """Strong marginal signal but a regime UNRELATED to it -> regime adds nothing
    (the trap the interaction test must avoid)."""
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.standard_normal(n)
    y = x + 0.3 * rng.standard_normal(n)         # strong marginal, regime-blind
    df = pd.DataFrame({"f": x, "_fwd_r5": y,
                       "rg_trend": rng.integers(0, 2, n),
                       "rg_vol": rng.integers(0, 3, n),
                       "rg_disaster": np.zeros(n, dtype=int)})
    res = probe(df, ["f"], ["_fwd_r5"], n_perm=120)
    v = _verdict(res)
    assert v["verdict"].startswith("NO-ADDED-INFORMATION")
