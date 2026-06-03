"""ID2 — identified LLM-contribution estimator.

On a leakage-free substrate (ID1 holds), the LLM's *incremental* predictive
content beyond a tabular baseline is the identified reasoning contribution. We
measure it as the partial rank correlation between the LLM signal and the
realised target, controlling for the baseline signal:

    contribution = corr( rank(LLM) ⟂ rank(baseline) ,  rank(target) ⟂ rank(baseline) )

i.e. correlate the parts of the LLM ranks and the target ranks that the baseline
does NOT already explain. Reported with a date-CLUSTERED bootstrap 95% CI.

CPU-only, numpy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["partial_rank_corr", "incremental_contribution"]


def _pctrank(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(a, method="average") / len(a)


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of y after OLS regression on x (both 1-D), incl. intercept."""
    x = x - x.mean()
    denom = (x * x).sum()
    beta = (x * (y - y.mean())).sum() / denom if denom > 0 else 0.0
    return (y - y.mean()) - beta * x


def partial_rank_corr(llm: np.ndarray, target: np.ndarray, baseline: np.ndarray) -> float:
    """Partial correlation of llm vs target controlling for baseline, on ranks."""
    lr, tr, br = _pctrank(llm), _pctrank(target), _pctrank(baseline)
    rl, rt = _residualize(lr, br), _residualize(tr, br)
    denom = np.linalg.norm(rl) * np.linalg.norm(rt)
    return float((rl * rt).sum() / denom) if denom > 0 else 0.0


def incremental_contribution(baseline: np.ndarray, llm: np.ndarray, target: np.ndarray,
                             dates: np.ndarray, n_boot: int = 1000, seed: int = 42,
                             ci: float = 95.0) -> dict:
    """Identified incremental LLM contribution (partial rank corr beyond baseline)
    with a date-clustered bootstrap CI."""
    baseline = np.asarray(baseline, float); llm = np.asarray(llm, float)
    target = np.asarray(target, float); dates = np.asarray(dates)
    point = partial_rank_corr(llm, target, baseline)

    uniq = np.unique(dates)
    by_date = {d: np.where(dates == d)[0] for d in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_date[d] for d in draw])
        vals.append(partial_rank_corr(llm[idx], target[idx], baseline[idx]))
    lo = float(np.percentile(vals, (100 - ci) / 2))
    hi = float(np.percentile(vals, 100 - (100 - ci) / 2))
    return {"mean": float(np.mean(vals)), "point": point, "lo": lo, "hi": hi,
            "n": int(len(baseline)), "n_dates": int(len(uniq))}
