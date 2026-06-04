"""TIM1 -- timing overlay + Sharpe decomposition. CPU, hermetic."""
import numpy as np
import pandas as pd

from src.onset.timing_overlay import timed_series, decompose_sharpe


def _data(seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    market = pd.Series(rng.normal(0.002, 0.02, n), index=[f"d{i}" for i in range(n)])
    # the worst 20% dates are the "disaster" dates -> mark them out-of-market
    bad = market < market.quantile(0.2)
    in_market = ~bad
    return market, in_market


def test_timed_series_zeroes_out_periods():
    market, in_market = _data()
    t = timed_series(market, in_market)
    assert (t[~in_market] == 0).all()
    assert (t[in_market] == market[in_market]).all()


def test_timing_raises_sharpe():
    market, in_market = _data()
    d = decompose_sharpe(market, in_market)
    assert d["timed_sharpe"] > d["buy_hold_sharpe"]      # avoiding bad dates helps
    assert d["incremental_timing"] > 0


def test_selection_adds_on_top():
    market, in_market = _data(1)
    sel = pd.Series(0.005, index=market.index)            # constant positive selection excess
    d = decompose_sharpe(market, in_market, selection_excess=sel)
    assert d["timed_plus_selected_sharpe"] >= d["timed_sharpe"]
    assert "incremental_selection" in d
