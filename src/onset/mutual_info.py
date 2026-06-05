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

__all__ = ["quantile_bins", "mutual_info", "conditional_mi", "perm_pvalue"]


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
    joint = np.zeros((kx, ky), dtype=float)
    np.add.at(joint, (cx, cy), 1.0)
    joint /= n
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
        obs = conditional_mi(x, y, z, bins)
        strata = [np.flatnonzero(z == s) for s in np.unique(z)]
        null = np.empty(n_perm)
        for i in range(n_perm):
            yp = y.copy()
            for idx in strata:
                yp[idx] = y[rng.permutation(idx)]
            null[i] = conditional_mi(x, yp, z, bins)
    else:
        obs = mutual_info(x, y, bins)
        null = np.empty(n_perm)
        for i in range(n_perm):
            null[i] = mutual_info(x, y[rng.permutation(x.size)], bins)
    p = (1.0 + float(np.sum(null >= obs))) / (1.0 + n_perm)
    return {"observed": float(obs), "null_mean": float(null.mean()),
            "p_value": float(p), "n_perm": int(n_perm)}
