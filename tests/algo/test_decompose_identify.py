"""NB2 — beta-timing vs alpha-selection identification. CPU, hermetic."""
import numpy as np

from src.identify.decompose_identify import selection_contribution, timing_contribution


def test_selection_positive_then_null():
    rng = np.random.default_rng(0)
    n = 600
    base = rng.normal(size=n); extra = rng.normal(size=n)
    neutral = extra + 0.3 * rng.normal(size=n)        # idiosyncratic target carries 'extra'
    llm = extra + 0.3 * rng.normal(size=n)
    dates = np.repeat(np.arange(n // 10), 10)
    r = selection_contribution(base, llm, neutral, dates, n_boot=300)
    assert r["lo"] > 0                                 # LLM adds idiosyncratic signal

    llm_redundant = base + 0.5 * rng.normal(size=n)
    r2 = selection_contribution(base, llm_redundant, neutral, dates, n_boot=300)
    assert r2["lo"] <= 0 <= r2["hi"]


def test_timing_positive_then_null():
    rng = np.random.default_rng(1)
    n_dates, per = 50, 10
    dates = np.repeat(np.arange(n_dates), per)
    market = rng.normal(size=n_dates)                  # per-date systematic move
    systematic = np.repeat(market, per)
    # LLM aggregate tracks the market move
    llm = np.repeat(market, per) + 0.3 * rng.normal(size=n_dates * per)
    r = timing_contribution(llm, systematic, dates, n_boot=300)
    assert r["lo"] > 0 and r["n_dates"] == n_dates

    llm_rand = rng.normal(size=n_dates * per)
    r2 = timing_contribution(llm_rand, systematic, dates, n_boot=300)
    assert r2["lo"] <= 0 <= r2["hi"]
