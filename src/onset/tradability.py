"""SGN1 -- decompose conditional information into TRADABLE vs SIGN-BLIND.

Mutual information (the onset-motif line) is SIGN-BLIND: I(X;Y|Z)>0 can come from
the conditional MEAN E[Y|X,Z] moving with X (DIRECTIONAL -> tradable long) or from
the conditional VARIANCE Var[Y|X,Z] moving with X (volatility/risk -> NOT directly
tradable long). These estimators measure the part MI cannot see:

  directionality : per-regime slope of E[Y|X] (rank corr of feature vs return) and
                   the sign-hit-rate of the feature about the return -- the
                   directional, long-tradable component.
  monotonicity   : per-regime conditional mean of Y across feature QUANTILE buckets
                   plus a monotonicity coefficient (rank corr of bucket index vs
                   bucket mean). Monotone => directly tradable; hump/U => not.
  variance_vs_mean: how much of the feature's predictive content is in E[Y|X]
                   (directional) vs Var[Y|X] (sign-blind).

CPU-only, numpy/scipy. SIGN-T1: conditional MEAN, never MI.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["directionality", "monotonicity", "variance_vs_mean"]


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def directionality(x: np.ndarray, y: np.ndarray,
                   z_states: np.ndarray | None = None) -> dict:
    """Per-regime directional slope (Spearman of feature vs return) + sign-hit-rate
    (fraction where sign of demeaned feature matches sign of demeaned return).
    Returns per-state dict and a sample-weighted overall value."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    z = (np.zeros(x.size, int) if z_states is None
         else np.asarray(z_states))
    per = {}
    tot_n, tot_slope, tot_hit = 0, 0.0, 0.0
    for s in np.unique(z):
        m = z == s
        xs, ys = x[m], y[m]
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if xs.size < 30:
            continue
        slope = _rank_corr(xs, ys)
        hit = float(np.mean(np.sign(xs - np.median(xs)) ==
                            np.sign(ys - np.median(ys))))
        per[int(s)] = {"n": int(xs.size), "slope": slope, "sign_hit": hit}
        tot_n += xs.size; tot_slope += slope * xs.size; tot_hit += hit * xs.size
    overall = {"slope": (tot_slope / tot_n if tot_n else 0.0),
               "sign_hit": (tot_hit / tot_n if tot_n else 0.0), "n": tot_n}
    return {"per_state": per, "overall": overall}


def monotonicity(x: np.ndarray, y: np.ndarray,
                 z_states: np.ndarray | None = None, q: int = 5) -> dict:
    """Per-regime conditional mean of y across q feature-quantile buckets, plus a
    monotonicity coefficient = rank corr of bucket index vs bucket mean (+1 fully
    monotone increasing, ~0 non-monotone/hump). Tradability needs |coef| high."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    z = (np.zeros(x.size, int) if z_states is None else np.asarray(z_states))
    per = {}
    tot_n, tot_coef = 0, 0.0
    for s in np.unique(z):
        m = z == s
        xs, ys = x[m], y[m]
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if xs.size < q * 20 or np.unique(xs).size < q:
            continue
        edges = np.unique(np.quantile(xs, np.linspace(0, 1, q + 1)))
        if edges.size < 3:
            continue
        bucket = np.clip(np.digitize(xs, edges[1:-1]), 0, edges.size - 2)
        means = np.array([ys[bucket == b].mean() for b in range(edges.size - 1)])
        coef = _rank_corr(np.arange(means.size), means)
        per[int(s)] = {"n": int(xs.size), "bucket_means": means.tolist(),
                       "mono_coef": coef,
                       "spread": float(means.max() - means.min())}
        tot_n += xs.size; tot_coef += coef * xs.size
    overall = {"mono_coef": (tot_coef / tot_n if tot_n else 0.0), "n": tot_n}
    return {"per_state": per, "overall": overall}


def variance_vs_mean(x: np.ndarray, y: np.ndarray,
                     z_states: np.ndarray | None = None, q: int = 5) -> dict:
    """Decompose the feature's predictive content into a MEAN component (does E[y]
    vary across feature buckets? -> directional) and a VARIANCE component (does
    Var[y] vary across buckets? -> sign-blind). Returns the across-bucket std of
    bucket means vs the across-bucket std of bucket stds, and the directional
    fraction mean_var / (mean_var + var_var). High fraction => tradable."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    z = (np.zeros(x.size, int) if z_states is None else np.asarray(z_states))
    mean_disp, var_disp, n_tot = 0.0, 0.0, 0
    for s in np.unique(z):
        m = z == s
        xs, ys = x[m], y[m]
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if xs.size < q * 20 or np.unique(xs).size < q:
            continue
        edges = np.unique(np.quantile(xs, np.linspace(0, 1, q + 1)))
        if edges.size < 3:
            continue
        bucket = np.clip(np.digitize(xs, edges[1:-1]), 0, edges.size - 2)
        bmeans = np.array([ys[bucket == b].mean() for b in range(edges.size - 1)])
        bstds = np.array([ys[bucket == b].std() for b in range(edges.size - 1)])
        mean_disp += np.std(bmeans) * xs.size
        var_disp += np.std(bstds) * xs.size
        n_tot += xs.size
    if n_tot == 0:
        return {"mean_dispersion": 0.0, "var_dispersion": 0.0,
                "directional_fraction": 0.0, "n": 0}
    md, vd = mean_disp / n_tot, var_disp / n_tot
    frac = md / (md + vd) if (md + vd) > 0 else 0.0
    return {"mean_dispersion": float(md), "var_dispersion": float(vd),
            "directional_fraction": float(frac), "n": n_tot}
