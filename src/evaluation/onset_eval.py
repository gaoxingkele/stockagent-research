"""T-005 — leakage-safe, cluster-robust evaluation for onset models.

Unifies the two methodological findings of the paper into one evaluation entry:
  * C3: date-CLUSTERED bootstrap CIs — anchors sharing a trade date are
    correlated, so resampling whole dates (not individual anchors) gives honest
    intervals. The naive anchor bootstrap understates uncertainty.
  * C5: a point-in-time guard that refuses any feature timestamped at or after
    the prediction target date (temporal-leakage prevention).

CPU-only, numpy/pandas.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

__all__ = ["clustered_bootstrap", "naive_bootstrap", "point_in_time_guard"]

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _ci(values, ci: float):
    lo = float(np.percentile(values, (100 - ci) / 2))
    hi = float(np.percentile(values, 100 - (100 - ci) / 2))
    return {"mean": float(np.mean(values)), "lo": lo, "hi": hi}


def clustered_bootstrap(metric_fn: MetricFn, preds, target, dates,
                        n_boot: int = 1000, seed: int = 42, ci: float = 95.0) -> dict:
    """Date-clustered bootstrap: resample whole trading days with replacement."""
    preds = np.asarray(preds); target = np.asarray(target); dates = np.asarray(dates)
    uniq = np.unique(dates)
    by_date = {d: np.where(dates == d)[0] for d in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_date[d] for d in draw])
        vals.append(metric_fn(preds[idx], target[idx]))
    return _ci(vals, ci)


def naive_bootstrap(metric_fn: MetricFn, preds, target,
                    n_boot: int = 1000, seed: int = 42, ci: float = 95.0) -> dict:
    """Anchor-independent bootstrap (the common, anti-conservative baseline)."""
    preds = np.asarray(preds); target = np.asarray(target)
    n = len(preds)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(preds[idx], target[idx]))
    return _ci(vals, ci)


def point_in_time_guard(feature_dates, target_date) -> None:
    """Raise ValueError if any feature timestamp is at or after the target date.

    This is the contribution-C5 leakage guard: a feature observed on or after the
    prediction date would leak the future.
    """
    fd = pd.to_datetime(np.asarray(feature_dates))
    td = pd.to_datetime(target_date)
    if (fd >= td).any():
        bad = fd[fd >= td]
        raise ValueError(
            f"point-in-time violation: {len(bad)} feature date(s) >= target {td.date()} "
            f"(e.g. {bad[0].date()}) — would leak the future."
        )
