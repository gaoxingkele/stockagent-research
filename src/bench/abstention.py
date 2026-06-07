"""BENCH3 -- two-level-uncertainty abstention gate (When Alpha Breaks, 2603.13252).

Abstain (hold cash) when EITHER model-prediction uncertainty OR regime-shift
instability exceeds a learned threshold. Thresholds are expanding-window quantiles
fit on PAST observations only (point-in-time, SIGN-A1) -- at date t we abstain on
the upper tail of uncertainty/instability as seen up to t-1, never using the
future. EXP-B applies this to our onset ranker.

CPU-only, numpy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["regime_instability", "abstain_mask"]


def regime_instability(states: np.ndarray, window: int = 5) -> np.ndarray:
    """Recent regime-switch rate: fraction of state changes in the trailing
    `window` (uses states <= t only -> point-in-time)."""
    s = np.asarray(states)
    T = len(s)
    switch = np.zeros(T, dtype=float)
    switch[1:] = (s[1:] != s[:-1]).astype(float)
    out = np.zeros(T, dtype=float)
    for t in range(T):
        lo = max(0, t - window + 1)
        out[t] = switch[lo:t + 1].mean()
    return out


def _expanding_quantile(x: np.ndarray, q: float, min_warmup: int) -> np.ndarray:
    """threshold[t] = quantile(x[:t], q); warmup -> +inf (do not abstain yet)."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    thr = np.full(T, np.inf)
    for t in range(T):
        if t >= min_warmup:
            thr[t] = np.quantile(x[:t], q)
    return thr


def abstain_mask(pred_uncertainty: np.ndarray, regime_instability: np.ndarray,
                 q_unc: float = 0.8, q_reg: float = 0.8,
                 min_warmup: int = 20) -> np.ndarray:
    """Per-date boolean: True = TRADE, False = abstain (cash). Abstains when either
    signal is in its upper-tail (above the expanding past q-quantile). During
    warmup (threshold = +inf) we trade by default."""
    u = np.asarray(pred_uncertainty, dtype=float)
    r = np.asarray(regime_instability, dtype=float)
    tu = _expanding_quantile(u, q_unc, min_warmup)
    tr = _expanding_quantile(r, q_reg, min_warmup)
    return (u <= tu) & (r <= tr)
