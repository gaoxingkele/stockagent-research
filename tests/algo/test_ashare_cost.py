"""COST1 -- realistic A-share cost model. CPU, hermetic."""
import numpy as np
import pandas as pd

from src.onset.ashare_cost import round_trip_cost, enterable, net_excess, DEFAULT_ROUND_TRIP


def test_round_trip_default_and_stamp_on_sell_only():
    assert abs(DEFAULT_ROUND_TRIP - 0.002) < 1e-9          # ~0.2%
    # removing the sell stamp lowers the round-trip by exactly stamp
    assert abs(round_trip_cost() - round_trip_cost(stamp_sell=0.0) - 0.0005) < 1e-12


def test_enterable_excludes_limit_up():
    df = pd.DataFrame({"pct_chg": [0.0, 5.0, 9.9, -3.0, 10.0]})
    m = enterable(df)
    assert list(m) == [True, True, False, True, False]


def test_net_excess_subtracts_cost():
    s = pd.Series([0.01, 0.02, -0.005])
    out = net_excess(s, round_trip=0.002)
    assert np.allclose(out.values, s.values - 0.002)
