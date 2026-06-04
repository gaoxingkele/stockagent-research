"""LO1 -- long-only top-K market-excess. CPU, hermetic."""
import numpy as np
import pandas as pd

from src.onset.long_only import long_only_excess, summarize_excess


def _df(seed, signal_strength):
    rng = np.random.default_rng(seed)
    n = 600
    dates = np.repeat(np.arange(n // 30), 30)
    ret = rng.normal(0, 0.03, n)
    sig = signal_strength * ret + rng.normal(0, 0.03, n)   # correlated with ret if strength>0
    return pd.DataFrame({"trade_date": dates.astype(str), "_fwd_r5": ret, "sig": sig})


def test_positive_excess_when_signal_informative():
    s = long_only_excess(_df(0, signal_strength=2.0), k_frac=0.2)
    r = summarize_excess(s, n_boot=300)
    assert r["mean_per_period"] > 0
    assert r["mean_ci95"][0] > 0                # significantly positive


def test_null_when_signal_random():
    s = long_only_excess(_df(1, signal_strength=0.0), k_frac=0.2)
    r = summarize_excess(s, n_boot=300)
    assert r["mean_ci95"][0] <= 0 <= r["mean_ci95"][1]


def test_cost_lowers_mean():
    s = long_only_excess(_df(0, signal_strength=2.0), k_frac=0.2)
    gross = summarize_excess(s, cost=0.0, n_boot=200)["mean_per_period"]
    net = summarize_excess(s, cost=0.002, n_boot=200)["mean_per_period"]
    assert abs((gross - net) - 0.002) < 1e-9
