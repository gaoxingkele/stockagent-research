"""MI1 -- mutual-information estimators with a within-stratum permutation null.

Binned (rank/quantile) estimators for the MARGINAL information I(X;Y) and the
CONDITIONAL information I(X;Y|Z) (Z discretized into states; MI estimated within
each state and weighted by P(z)).

CRITICAL (SIGN-M1): binned MI is positively BIASED in finite samples, so a
positive raw MI is NOT evidence. Significance comes ONLY from a PERMUTATION null:
shuffle Y -- WITHIN each Z-stratum for the conditional test -- which destroys the
(conditional) dependence while preserving the marginals and the bias. The p-value
is the fraction of permuted MIs >= the observed MI.

CPU-only, numpy. nats.
"""
from __future__ import annotations

import numpy as np

__all__ = ["quantile_bins", "mutual_info", "conditional_mi", "perm_pvalue",
           "interaction_pvalue"]


def quantile_bins(x: np.ndarray, bins: int = 8) -> np.ndarray:
    """Discretize x into `bins` roughly equal-frequency bins -> integer codes.

    Ties collapse: distinct edges only, so the effective bin count may be < bins
    when x is heavily tied (e.g. a near-constant column). Robust to outliers."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=int)
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if edges.size <= 2:                      # degenerate / constant -> single bin
        return np.zeros(x.size, dtype=int)
    # interior edges only; np.digitize maps to 0..len(edges)-2
    codes = np.digitize(x, edges[1:-1], right=False)
    return codes.astype(int)


def _mi_from_codes(cx: np.ndarray, cy: np.ndarray) -> float:
    """Plug-in MI (nats) from two integer-code arrays via the joint histogram."""
    n = cx.size
    if n == 0:
        return 0.0
    kx = int(cx.max()) + 1
    ky = int(cy.max()) + 1
    if kx < 2 or ky < 2:
        return 0.0
    # bincount on the flattened (cx,cy) index -- much faster than np.add.at
    joint = np.bincount(cx * ky + cy, minlength=kx * ky).astype(float)
    joint = joint.reshape(kx, ky) / n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px * py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))


def mutual_info(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """Marginal MI I(X;Y) in nats, quantile-binned. Positively biased (SIGN-M1)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return _mi_from_codes(quantile_bins(x, bins), quantile_bins(y, bins))


def conditional_mi(x: np.ndarray, y: np.ndarray, z_states: np.ndarray,
                   bins: int = 8) -> float:
    """Conditional MI I(X;Y|Z) = sum_z P(z) * MI(X,Y | Z=z), quantile-binned within
    each stratum. z_states is an integer state per sample."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z_states)
    n = x.size
    if n == 0:
        return 0.0
    total = 0.0
    for s in np.unique(z):
        m = z == s
        ns = int(m.sum())
        if ns < bins:                        # too few to bin meaningfully -> 0
            continue
        cmi = _mi_from_codes(quantile_bins(x[m], bins), quantile_bins(y[m], bins))
        total += (ns / n) * cmi
    return float(total)


def perm_pvalue(x: np.ndarray, y: np.ndarray, z_states: np.ndarray | None = None,
                n_perm: int = 500, conditional: bool = True, bins: int = 8,
                seed: int = 0) -> dict:
    """Permutation p-value for (conditional) MI.

    conditional=True : shuffle Y WITHIN each Z-stratum (breaks I(X;Y|Z), keeps
                       marginals + bias) and compare to the observed conditional MI.
    conditional=False: shuffle Y globally and compare to the observed marginal MI.

    Returns {observed, null_mean, p_value, n_perm}. p = (1 + #{perm >= obs}) /
    (1 + n_perm)  (never zero -- the standard unbiased permutation estimate)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if conditional:
        z = np.asarray(z_states)
        n = x.size
        # Precompute, per stratum, the x/y bin codes ONCE (x never changes; the
        # null only permutes y, and binning the SAME y values in a shuffled order
        # gives the same codes in shuffled order -> we permute codes, not values).
        strata = []
        for s in np.unique(z):
            idx = np.flatnonzero(z == s)
            if idx.size < bins:
                continue
            cx = quantile_bins(x[idx], bins)
            cy = quantile_bins(y[idx], bins)
            strata.append((idx.size, cx, cy))
        obs = sum((ns / n) * _mi_from_codes(cx, cy) for ns, cx, cy in strata)
        null = np.empty(n_perm)
        for i in range(n_perm):
            null[i] = sum((ns / n) * _mi_from_codes(cx, cy[rng.permutation(ns)])
                          for ns, cx, cy in strata)
    else:
        cx = quantile_bins(x, bins)
        cy = quantile_bins(y, bins)
        obs = _mi_from_codes(cx, cy)
        null = np.empty(n_perm)
        for i in range(n_perm):
            null[i] = _mi_from_codes(cx, cy[rng.permutation(cx.size)])
    p = (1.0 + float(np.sum(null >= obs))) / (1.0 + n_perm)
    return {"observed": float(obs), "null_mean": float(null.mean()),
            "p_value": float(p), "n_perm": int(n_perm)}


def interaction_pvalue(x: np.ndarray, y: np.ndarray, z_states: np.ndarray,
                       n_perm: int = 500, bins: int = 8, seed: int = 0) -> dict:
    """The DECISIVE motif test: does the regime Z ADD information?

    Tests interaction information II = I(X;Y|Z) - I(X;Y) > 0 against a null where
    Z is PERMUTED across samples (X,Y pairs kept intact). Permuting Z destroys any
    genuine three-way X-Y-Z structure while preserving the marginal X-Y relation
    AND the per-stratum binning bias -- so a positive result means conditioning on
    the REAL regime buys more than a random regime label of the same granularity.

    Returns observed conditional MI, marginal MI, interaction (cond-marg), the
    null distribution of conditional MI under permuted Z, and p_value =
    P(cond_MI(perm Z) >= observed cond_MI). p<0.05 => the regime adds real info.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z_states)
    n = x.size
    cx = quantile_bins(x, bins)          # GLOBAL codes (fixed), so permuting Z is cheap
    cy = quantile_bins(y, bins)
    marg = _mi_from_codes(cx, cy)

    def _cond(zz):
        tot = 0.0
        for s in np.unique(zz):
            m = zz == s
            ns = int(m.sum())
            if ns < bins:
                continue
            tot += (ns / n) * _mi_from_codes(cx[m], cy[m])
        return tot

    cond_obs = _cond(z)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _cond(z[rng.permutation(n)])
    p = (1.0 + float(np.sum(null >= cond_obs))) / (1.0 + n_perm)
    return {"cond_mi": float(cond_obs), "marg_mi": float(marg),
            "interaction": float(cond_obs - marg),
            "null_cond_mean": float(null.mean()), "p_value": float(p),
            "n_perm": int(n_perm)}
