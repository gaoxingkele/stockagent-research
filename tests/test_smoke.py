"""Smoke tests: verify imports, basic label generation, and metric computation."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def make_synthetic_panel(n_stocks: int = 20, n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic synthetic panel for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stocks):
        price = 100.0
        for d in range(n_days):
            ret = rng.normal(0.0005, 0.02)
            price *= (1 + ret)
            rows.append({
                "ts_code": f"{s:06d}.SH",
                "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "close": price,
            })
    df = pd.DataFrame(rows)
    df["trade_date"] = df["trade_date"].dt.strftime("%Y%m%d")
    return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


# ---------- Label tests ----------

def test_fixed_horizon_label_basic():
    from src.labels.fixed_horizon import fixed_horizon_label

    df = make_synthetic_panel()
    y = fixed_horizon_label(df, horizon=5, threshold_up=0.02)
    assert len(y) == len(df)
    # Last 5 days of each stock should be -127 (no forward data)
    last_rows = df.groupby("ts_code").tail(5)
    assert (y.loc[last_rows.index] == -127).all()
    # Other values must be in {-1, 0, 1}
    valid = y[y != -127]
    assert valid.isin([-1, 0, 1]).all()


def test_fixed_window_pwc_label_basic():
    from src.labels.fixed_window_pwc import fixed_window_pwc_label

    df = make_synthetic_panel()
    y = fixed_window_pwc_label(df)
    assert len(y) == len(df)
    valid = y[y != -127]
    assert valid.isin([-1, 0, 1]).all()
    # PWC should be more sparse than FH (most samples = 0)
    onset_rate = (valid != 0).mean()
    assert onset_rate < 0.5, f"PWC onset rate too high: {onset_rate}"


# ---------- Metric tests ----------

def test_rank_ic_basic():
    from src.evaluation.metrics import cross_sectional_rank_ic, information_ratio

    rng = np.random.default_rng(0)
    n = 100
    dates = pd.Series(np.repeat(["20240101", "20240102", "20240103"], n // 3 + 1)[:n])
    target = pd.Series(rng.normal(size=n))
    # Perfect predictor
    pred = target + rng.normal(0, 0.01, size=n)
    ic = cross_sectional_rank_ic(pred, target, dates)
    assert ic.mean() > 0.8

    ir = information_ratio(ic)
    assert np.isfinite(ir)


def test_topk_return():
    from src.evaluation.metrics import topk_return

    n = 30
    dates = pd.Series(["20240101"] * n)
    pred = pd.Series(range(n))
    target = pd.Series(range(n))  # perfectly aligned
    tk = topk_return(pred, target, dates, k=5)
    # Top 5 are predictions 25..29, mean target = 27
    assert abs(tk.iloc[0] - 27.0) < 1e-6


# ---------- Model tests ----------

def test_lgbm_train_predict_smoke():
    from src.models.lgbm_baseline import LGBMConfig, train, predict_signal

    rng = np.random.default_rng(42)
    n, d = 500, 10
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    # Make label correlated with first feature
    y_raw = (X["f0"] + rng.normal(0, 0.5, size=n))
    y = pd.Series(np.where(y_raw > 0.5, 1, np.where(y_raw < -0.5, -1, 0)))

    cfg = LGBMConfig(num_boost_round=20, early_stopping_rounds=5)
    booster = train(X[:400], y[:400], X[400:], y[400:], cfg=cfg)
    signal = predict_signal(booster, X[400:])
    assert len(signal) == 100
    assert np.isfinite(signal).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
