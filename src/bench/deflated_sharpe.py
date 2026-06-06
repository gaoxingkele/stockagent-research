"""BENCH2 -- Probabilistic & Deflated Sharpe Ratio (Bailey & Lopez de Prado).

Corrects an observed Sharpe for (i) non-normal returns (skew, kurtosis), (ii)
sample length, and (iii) the number of strategy variants tried (multiple testing).
Required by SIGN-B1: a benchmark arm that wins on raw Sharpe but fails the DSR is
not a win.

Conventions: pass the ANNUALIZED Sharpe plus `periods_per_year`; it is converted
to a per-period Sharpe internally (the PSR/DSR formulas operate on per-period SR).
skew/kurtosis are of the per-period returns; kurtosis is non-excess (normal = 3).
Pure numpy/scipy.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["probabilistic_sharpe", "expected_max_sharpe", "deflated_sharpe"]

_EULER = 0.5772156649015329


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int,
                         skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """PSR: P(true SR > sr_benchmark) given the estimate sr (per-period units)."""
    if n_obs < 2:
        return float("nan")
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / denom))


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Expected maximum of n_trials independent Sharpe estimates with cross-trial
    variance var_sharpe (per-period units) -- the SR0 benchmark for the DSR."""
    if n_trials <= 1 or var_sharpe <= 0:
        return 0.0
    sd = np.sqrt(var_sharpe)
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - _EULER) * z1 + _EULER * z2))


def deflated_sharpe(sharpe: float, n_obs: int, skew: float = 0.0,
                    kurtosis: float = 3.0, n_trials: int = 1,
                    var_sharpe: float | None = None,
                    trial_sharpes=None, periods_per_year: float | None = None) -> dict:
    """Deflated Sharpe Ratio = PSR evaluated against the expected-max-Sharpe SR0.

    Returns dsr (deflated, multiple-testing-corrected), psr (vs 0), the per-period
    SR, the SR0 threshold, and the cross-trial Sharpe variance used."""
    sr = sharpe / np.sqrt(periods_per_year) if periods_per_year else float(sharpe)
    if var_sharpe is None:
        if trial_sharpes is not None and len(trial_sharpes) > 1:
            ts = np.asarray(trial_sharpes, dtype=float)
            if periods_per_year:
                ts = ts / np.sqrt(periods_per_year)
            var_sharpe = float(np.var(ts, ddof=1))
        else:
            var_sharpe = 0.0
    sr0 = expected_max_sharpe(n_trials, var_sharpe)
    return {"dsr": probabilistic_sharpe(sr, sr0, n_obs, skew, kurtosis),
            "psr": probabilistic_sharpe(sr, 0.0, n_obs, skew, kurtosis),
            "sr_per_period": sr, "sr0_threshold": float(sr0),
            "var_sharpe": float(var_sharpe), "n_trials": int(n_trials),
            "n_obs": int(n_obs)}
