"""BENCH1 -- point-in-time rolling Gaussian-HMM market regime states.

Reimplementation of the rolling-HMM regime component of Regime-Aware LightGBM
(MDPI Electronics 2026) on our data, so EXP-A can condition LightGBM on HMM
regimes exactly as src/onset/regimes.py provides trend/vol/disaster.

POINT-IN-TIME (SIGN-A1): the regime at date t is decoded from observations <= t
only. The HMM is refit on an expanding PAST window every `refit_every` steps; the
state at t is the last element of a Viterbi decode of X[:t+1]. States are
canonicalized (sorted by the mean of feature 0) so label identity is stable across
refits. CPU-only.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["rolling_hmm_states"]


def _fit_canonical(X_past: np.ndarray, n_states: int, seed: int):
    from hmmlearn.hmm import GaussianHMM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=50, random_state=seed)
        m.fit(X_past)
    # canonical order: ascending mean of feature 0 -> state 0 = lowest-return regime
    order = np.argsort(m.means_[:, 0])
    remap = np.empty(n_states, dtype=int)
    remap[order] = np.arange(n_states)
    return m, remap


def rolling_hmm_states(market_features: np.ndarray, n_states: int = 3,
                       min_train: int = 120, refit_every: int = 20,
                       seed: int = 0) -> np.ndarray:
    """Per-date integer regime state, point-in-time.

    market_features: (T,) or (T, d) array of per-date market observables
    (e.g. market return, rolling vol). Returns an int array of length T; the
    warmup region (< min_train) is state 0."""
    X = np.asarray(market_features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T = len(X)
    states = np.zeros(T, dtype=int)
    model = None
    remap = None
    for t in range(T):
        if t < min_train:
            continue
        if model is None or (t % refit_every == 0):
            try:
                model, remap = _fit_canonical(X[:t], n_states, seed)
            except Exception:
                pass
        if model is not None:
            try:
                raw = model.predict(X[:t + 1])[-1]   # obs <= t only
                states[t] = int(remap[raw])
            except Exception:
                states[t] = states[t - 1]
        else:
            states[t] = states[t - 1]
    return states
