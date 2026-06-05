"""MI1 -- hermetic: X informs Y ONLY given Z (conditional info, ~0 marginal)."""
import numpy as np

from src.onset.mutual_info import (mutual_info, conditional_mi, perm_pvalue,
                                   quantile_bins, interaction_pvalue)


def _xor_like(n=4000, seed=1):
    """XOR-style: x,y are constant-magnitude signs (+/-1) so NO magnitude info
    leaks. y_sign = x_sign * z_sign. Marginally z_sign is +/-1 each w.p. 1/2, so
    y_sign is independent of x_sign -> I(X;Y) ~ 0. GIVEN Z, y_sign = x_sign*z_sign
    is deterministic -> conditional info is full."""
    rng = np.random.default_rng(seed)
    xs = rng.choice([-1.0, 1.0], n)
    z = rng.integers(0, 2, n)               # state 0 / 1
    zs = np.where(z == 0, -1.0, 1.0)
    ys = xs * zs
    x = xs + 0.05 * rng.standard_normal(n)
    y = ys + 0.05 * rng.standard_normal(n)
    return x, y, z


def test_quantile_bins_equal_frequency():
    codes = quantile_bins(np.arange(100.0), bins=4)
    counts = np.bincount(codes)
    assert codes.max() == 3
    assert counts.min() >= 20 and counts.max() <= 30


def test_quantile_bins_constant_single_bin():
    assert quantile_bins(np.ones(50), bins=8).max() == 0


def test_conditional_exceeds_marginal():
    x, y, z = _xor_like()
    marg = mutual_info(x, y)
    cond = conditional_mi(x, y, z)
    assert cond > marg + 0.2          # conditional info is large, marginal ~0
    assert marg < 0.05


def test_permutation_null_distinguishes():
    x, y, z = _xor_like()
    cond = perm_pvalue(x, y, z, n_perm=200, conditional=True)
    marg = perm_pvalue(x, y, z, n_perm=200, conditional=False)
    assert cond["p_value"] < 0.02     # real conditional dependence -> significant
    assert marg["p_value"] > 0.10     # no marginal dependence -> not significant


def test_permutation_null_kills_noise():
    """When X is pure noise wrt Y even given Z, conditional p-value is large."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(3000)
    y = rng.standard_normal(3000)
    z = rng.integers(0, 3, 3000)
    r = perm_pvalue(x, y, z, n_perm=200, conditional=True)
    assert r["p_value"] > 0.05


def test_interaction_detects_regime_added_info():
    """When the regime GENUINELY adds info (XOR: Z flips the X->Y relation), the
    Z-permutation interaction test is significant and interaction > 0."""
    x, y, z = _xor_like()
    r = interaction_pvalue(x, y, z, n_perm=200)
    assert r["interaction"] > 0
    assert r["p_value"] < 0.02


def test_interaction_null_when_regime_irrelevant():
    """When Z is independent of the (real) X->Y relation, the interaction test is
    NOT significant -- conditioning on a random regime adds nothing. This is the
    trap: I(X;Y|Z)>0 trivially, but Z adds no information."""
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.standard_normal(n)
    y = x + 0.3 * rng.standard_normal(n)        # strong marginal X->Y
    z = rng.integers(0, 2, n)                    # regime unrelated to the relation
    r = interaction_pvalue(x, y, z, n_perm=200)
    assert r["p_value"] > 0.05
