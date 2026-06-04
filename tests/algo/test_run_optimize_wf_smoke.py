"""OPT1 -- hermetic smoke for walk-forward selection."""
from src.onset.run_optimize_wf import select_best


def test_select_best_picks_train_argmax():
    train_means = {"a": 0.1, "b": 0.5, "c": -0.2}
    test_results = {"a": {"annualized_sharpe": 1.0}, "b": {"annualized_sharpe": -0.3}, "c": {"annualized_sharpe": 0.2}}
    r = select_best(train_means, test_results)
    assert r["selected_config"] == "b"                  # best on TRAIN
    assert r["test_oos"]["annualized_sharpe"] == -0.3   # judged on TEST (here OOS is bad)
