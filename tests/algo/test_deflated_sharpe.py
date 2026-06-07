"""BENCH2 -- hermetic: DSR penalizes multiple testing; strong/marginal cases."""
from src.bench.deflated_sharpe import (probabilistic_sharpe, expected_max_sharpe,
                                       deflated_sharpe)

PPY = 50.4   # PERIODS_PER_YEAR (5-day periods), matches long_only


def test_expected_max_grows_with_trials():
    assert expected_max_sharpe(50, 0.01) > expected_max_sharpe(5, 0.01) > 0.0
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_more_trials_lower_dsr_same_sharpe():
    a = deflated_sharpe(1.0, n_obs=200, n_trials=1, var_sharpe=0.01, periods_per_year=PPY)
    b = deflated_sharpe(1.0, n_obs=200, n_trials=50, var_sharpe=0.01, periods_per_year=PPY)
    assert b["sr0_threshold"] > a["sr0_threshold"]
    assert b["dsr"] < a["dsr"]


def test_strong_single_trial_significant():
    r = deflated_sharpe(2.0, n_obs=250, n_trials=1, periods_per_year=PPY)
    assert r["dsr"] > 0.95          # genuine strong Sharpe survives


def test_marginal_many_trials_deflates():
    r = deflated_sharpe(0.8, n_obs=200, skew=0.0, kurtosis=3.0,
                        n_trials=50, var_sharpe=0.01, periods_per_year=PPY)
    assert r["psr"] > 0.5           # looks okay before correction
    assert r["dsr"] < 0.5           # collapses after multiple-testing correction


def test_psr_monotone_in_sharpe():
    lo = probabilistic_sharpe(0.05, 0.0, 200)
    hi = probabilistic_sharpe(0.30, 0.0, 200)
    assert hi > lo
