"""SGN1 -- hermetic: directional vs sign-blind information (what MI can't see)."""
import numpy as np

from src.onset.tradability import directionality, monotonicity, variance_vs_mean
from src.onset.mutual_info import mutual_info


def _directional(n=6000, seed=0):
    """E[y|x] monotone in x; regime flips the sign. Tradable within regime."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.integers(0, 2, n)
    sign = np.where(z == 0, -1.0, 1.0)
    y = sign * 0.5 * x + rng.standard_normal(n)      # mean moves with x
    return x, y, z


def _sign_blind(n=6000, seed=1):
    """Var[y|x] depends on |x| but E[y|x]=0. MI>0 but NOT directionally tradable."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.integers(0, 2, n)
    y = (0.3 + np.abs(x)) * rng.standard_normal(n)   # variance moves with x, mean 0
    return x, y, z


def test_directional_scores_high():
    x, y, z = _directional()
    d = directionality(x, y, z)
    m = monotonicity(x, y, z)
    # within each regime the slope is strong (sign differs by regime)
    assert all(abs(v["slope"]) > 0.15 for v in d["per_state"].values())
    assert all(abs(v["mono_coef"]) > 0.7 for v in m["per_state"].values())


def test_sign_blind_has_mi_but_no_direction():
    x, y, z = _sign_blind()
    # MI sees the dependence...
    assert mutual_info(x, y) > 0.01
    # ...but directionality + monotonicity are ~0 (no tradable direction)
    d = directionality(x, y, z)
    m = monotonicity(x, y, z)
    assert abs(d["overall"]["slope"]) < 0.05
    assert abs(m["overall"]["mono_coef"]) < 0.5


def test_variance_vs_mean_splits():
    xd, yd, zd = _directional()
    xs, ys, zs = _sign_blind()
    fd = variance_vs_mean(xd, yd, zd)["directional_fraction"]
    fs = variance_vs_mean(xs, ys, zs)["directional_fraction"]
    # directional data is more mean-driven than sign-blind data
    assert fd > fs
    assert fs < 0.5
