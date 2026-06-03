"""ID2 — identified LLM-contribution estimator. CPU, hermetic."""
import numpy as np

from src.identify.contribution import incremental_contribution, partial_rank_corr


def test_positive_when_llm_adds_signal_beyond_baseline():
    rng = np.random.default_rng(0)
    n = 600
    base_sig = rng.normal(size=n)
    llm_extra = rng.normal(size=n)
    target = base_sig + llm_extra + 0.3 * rng.normal(size=n)
    baseline = base_sig + 0.3 * rng.normal(size=n)
    llm = llm_extra + 0.3 * rng.normal(size=n)        # carries info NOT in baseline
    dates = np.repeat(np.arange(n // 10), 10)
    r = incremental_contribution(baseline, llm, target, dates, n_boot=300)
    assert r["mean"] > 0
    assert r["lo"] > 0                                 # significantly positive


def test_null_when_llm_is_baseline_plus_noise():
    rng = np.random.default_rng(1)
    n = 600
    base_sig = rng.normal(size=n)
    target = base_sig + 0.3 * rng.normal(size=n)
    baseline = base_sig + 0.3 * rng.normal(size=n)
    llm = baseline + 0.5 * rng.normal(size=n)          # no info beyond baseline
    dates = np.repeat(np.arange(n // 10), 10)
    r = incremental_contribution(baseline, llm, target, dates, n_boot=300)
    assert r["lo"] <= 0 <= r["hi"]                     # CI spans zero


def test_partial_rank_corr_finite():
    rng = np.random.default_rng(2)
    a, b, c = (rng.normal(size=50) for _ in range(3))
    v = partial_rank_corr(a, b, c)
    assert np.isfinite(v)
