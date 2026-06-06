"""EXP-B -- hermetic smoke: evaluate runs 3 arms; verdict logic."""
import numpy as np
import pandas as pd

from src.bench.run_expb import evaluate, _verdict


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    dates = [f"2024{1+i//20:02d}{1+i%20:02d}" for i in range(60)]
    rows = []
    for d in dates:
        for s in range(40):
            rows.append({"ts_code": f"{s:03d}.SZ", "trade_date": d,
                         "onset_score": rng.standard_normal(),
                         "_fwd_r5": rng.standard_normal() * 0.01})
    df = pd.DataFrame(rows)
    trend_up = pd.Series({d: (i % 3 != 0) for i, d in enumerate(dates)})
    return df, trend_up


def test_evaluate_three_arms():
    df, trend_up = _panel()
    res = evaluate(df, trend_up)
    assert set(res["pooled"]) == {"always", "trend", "abstain"}
    for m in res["pooled"].values():
        assert "annualized_sharpe" in m and "deflated" in m
    # abstain trades a subset of always
    assert res["traded_frac"]["abstain"] <= res["traded_frac"]["always"]


def test_verdict_no_rescue_when_negative():
    res = {"pooled": {"abstain": {"mean_ci95": [-0.003, 0.002],
                                  "deflated": {"dsr": 0.2}}}}
    assert _verdict(res)["abstention_rescues"] is False
