"""TRD2 -- hermetic smoke: backtest machinery + survivor/verdict logic."""
import numpy as np
import pandas as pd

from src.onset.run_regime_gate_bt import _bt_one


def _panel(edge: bool, seed=0):
    """Multi-year panel; in trend-up dates a feature has a directional edge if
    edge=True. Returns df + in_regime."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = []
    for yr in ("2022", "2023", "2024"):
        for i in range(40):
            dates.append(f"{yr}{1+i//20:02d}{1+i%20:02d}")
    up = {d: (i % 2 == 0) for i, d in enumerate(dates)}
    for d in dates:
        for s in range(40):
            f = rng.standard_normal()
            if edge and up[d]:
                r = 0.03 * f + 0.01 * rng.standard_normal()
            else:
                r = 0.01 * rng.standard_normal()
            rows.append({"ts_code": f"{s:03d}.SZ", "trade_date": d,
                         "feat": f, "_fwd_r5": r})
    df = pd.DataFrame(rows)
    in_regime = pd.Series({d: up[d] for d in dates})
    return df, in_regime


def test_strong_edge_survives_cross_period():
    df, in_regime = _panel(edge=True)
    r = _bt_one(df, "feat", in_regime)
    assert r["n_pos_years"] >= 2          # positive most years
    assert r["pooled_gated"]["annualized_sharpe"] > 0


def test_no_edge_does_not_survive():
    df, in_regime = _panel(edge=False)
    r = _bt_one(df, "feat", in_regime)
    # pure noise: should not be reliably positive across years
    assert r["n_pos_years"] <= 1
