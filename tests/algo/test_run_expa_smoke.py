"""EXP-A -- hermetic smoke: arm metrics + the 3-arm pipeline run end-to-end."""
import numpy as np
import pandas as pd

from src.bench.run_expa import arm_metrics, _verdict


def test_arm_metrics_keys():
    rng = np.random.default_rng(0)
    ex = pd.Series(rng.standard_normal(80) * 0.01,
                   index=[f"d{i:03d}" for i in range(80)])
    m = arm_metrics(ex, n_trials=3)
    assert "annualized_sharpe" in m and "deflated" in m
    assert "dsr" in m["deflated"] and "mean_ci95" in m


def test_verdict_all_subcost_when_negative():
    res = {"pooled": {
        "plain": {"mean_ci95": [-0.004, -0.001], "deflated": {"dsr": 0.1}},
        "hmm": {"mean_ci95": [-0.003, 0.002], "deflated": {"dsr": 0.3}},
        "trend": {"mean_ci95": [-0.002, 0.003], "deflated": {"dsr": 0.4}}}}
    v = _verdict(res)
    assert v["survivors"] == []
    assert "SUB-COST" in v["verdict"]


def test_verdict_survivor_requires_ci_and_dsr():
    res = {"pooled": {
        "plain": {"mean_ci95": [0.001, 0.01], "deflated": {"dsr": 0.99}},
        "hmm": {"mean_ci95": [0.001, 0.01], "deflated": {"dsr": 0.80}},  # CI ok, DSR fails
        "trend": {"mean_ci95": [-0.001, 0.01], "deflated": {"dsr": 0.99}}}}  # DSR ok, CI fails
    v = _verdict(res)
    assert v["survivors"] == ["plain"]
