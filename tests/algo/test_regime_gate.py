"""RGT1 -- hermetic: gate is non-zero only in-regime; directional edge -> profit."""
import numpy as np
import pandas as pd

from src.onset.regime_gate import regime_gated_excess, gated_vs_ungated


def _panel(seed=0):
    """30 stocks x 40 dates. In 'up' dates, feature predicts higher return (a real
    directional edge); in 'down' dates the feature is pure noise wrt return."""
    rng = np.random.default_rng(seed)
    dates = [f"2024{1+i//20:02d}{1+i%20:02d}" for i in range(40)]
    up = {d: (i % 2 == 0) for i, d in enumerate(dates)}   # alternate regime
    rows = []
    for d in dates:
        for s in range(30):
            f = rng.standard_normal()
            if up[d]:
                r = 0.02 * f + 0.01 * rng.standard_normal()   # edge: high f -> high r
            else:
                r = 0.01 * rng.standard_normal()              # no edge
            rows.append({"ts_code": f"{s:03d}.SZ", "trade_date": d,
                         "feat": f, "_fwd_r5": r})
    df = pd.DataFrame(rows)
    in_regime = pd.Series({d: up[d] for d in dates})
    return df, in_regime, dates, up


def test_gate_zero_out_of_regime():
    df, in_regime, dates, up = _panel()
    net = regime_gated_excess(df, "feat", in_regime, k_frac=0.2, cost=0.0)
    for d in net.index:
        if not up[d]:
            assert net.loc[d] == 0.0          # cash when out of regime


def test_directional_edge_is_profitable_gross():
    df, in_regime, dates, up = _panel()
    net = regime_gated_excess(df, "feat", in_regime, k_frac=0.2, cost=0.0)
    in_dates = [d for d in net.index if up[d]]
    # on traded dates the top-K-by-feature beats the cross-section on average
    assert np.mean([net.loc[d] for d in in_dates]) > 0


def test_gated_vs_ungated_shapes():
    df, in_regime, dates, up = _panel()
    out = gated_vs_ungated(df, "feat", in_regime, k_frac=0.2, cost=0.002)
    assert 0.0 < out["in_regime_frac"] < 1.0
    assert len(out["gated"]) == len(out["ungated"])
    # gated sits in cash half the time -> fewer nonzero days than ungated
    assert (out["gated"] != 0).sum() <= (out["ungated"] != 0).sum()
