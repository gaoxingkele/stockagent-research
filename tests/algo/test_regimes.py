"""REG1 -- hermetic: regime states are point-in-time and take expected values."""
import numpy as np
import pandas as pd

from src.onset.regimes import (trend_state, vol_state, regime_states,
                              map_states_to_rows)


def test_trend_state_point_in_time_and_values():
    # up for first half, down for second half
    r = pd.Series([0.02] * 10 + [-0.02] * 10,
                  index=[f"d{i:02d}" for i in range(20)])
    ts = trend_state(r, lookback=2)
    # first bar has no past -> default up(1); the down regime shows up AFTER the
    # trend actually turns (shifted), never at/before it -> point-in-time.
    assert ts.iloc[0] == 1
    assert ts.iloc[-1] == 0
    # the state at the turn date does NOT yet see the future down move
    assert ts.iloc[10] == 1


def test_vol_state_terciles():
    rng = np.random.default_rng(0)
    # rising volatility over time
    r = pd.Series(rng.standard_normal(300) * np.linspace(0.001, 0.05, 300),
                  index=[f"d{i:03d}" for i in range(300)])
    vs = vol_state(r, lookback=8)
    assert set(np.unique(vs)).issubset({0, 1, 2})
    # late (high-vol) dates skew to state 2 more than early dates
    assert vs.iloc[-50:].mean() > vs.iloc[:50].mean()


def test_regime_states_and_mapping():
    rng = np.random.default_rng(1)
    dates = [f"2024{m:02d}{d:02d}" for m in range(1, 4) for d in range(1, 21)]
    rows = []
    for code in ("000001.SZ", "000002.SZ"):
        for dt in dates:
            rows.append({"ts_code": code, "trade_date": dt,
                         "pct_chg": rng.standard_normal(), "amount": 1e6,
                         "_fwd_r5": rng.standard_normal() * 0.01})
    df = pd.DataFrame(rows)
    st = regime_states(df)
    assert {"trend", "vol", "disaster"}.issubset(st.columns)
    assert st["trend"].isin([0, 1]).all()
    assert st["disaster"].isin([0, 1]).all()
    mapped = map_states_to_rows(st["trend"], df["trade_date"])
    assert mapped.shape[0] == len(df)
    assert set(np.unique(mapped)).issubset({0, 1})
