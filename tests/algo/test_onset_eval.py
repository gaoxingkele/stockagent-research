"""T-005 — leakage-safe, cluster-robust evaluator. CPU, hermetic."""
import numpy as np
import pytest

from src.evaluation.onset_eval import clustered_bootstrap, naive_bootstrap, point_in_time_guard


def _accuracy(p, t):
    return float((p == t).mean())


def _date_correlated_data(seed=0):
    rng = np.random.default_rng(seed)
    n_dates, per = 20, 10
    dates = np.repeat(np.arange(n_dates), per)
    target = rng.integers(0, 2, n_dates * per)
    # correctness is constant within a date -> strong within-date correlation
    date_good = rng.random(n_dates) > 0.4
    correct = np.repeat(date_good, per)
    preds = np.where(correct, target, 1 - target)
    return preds, target, dates


def test_clustered_ci_wider_than_naive():
    preds, target, dates = _date_correlated_data()
    clustered = clustered_bootstrap(_accuracy, preds, target, dates, n_boot=500)
    naive = naive_bootstrap(_accuracy, preds, target, n_boot=500)
    w_clustered = clustered["hi"] - clustered["lo"]
    w_naive = naive["hi"] - naive["lo"]
    assert w_clustered > w_naive


def test_pit_guard_passes_when_features_precede_target():
    point_in_time_guard(["2020-01-01", "2020-01-02"], "2020-01-03")  # no raise


def test_pit_guard_raises_on_leaked_feature():
    with pytest.raises(ValueError):
        point_in_time_guard(["2020-01-02", "2020-01-03"], "2020-01-03")  # equal date leaks
