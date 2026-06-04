"""K1 -- candle geometry + relative-position features. CPU, hermetic."""
import numpy as np
import pandas as pd

from src.onset.candle_geometry import candle_features, FEATURE_COLS


def _ohlcv(seed=0, n=40):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    openp = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({
        "trade_date": [f"2025{i:04d}" for i in range(n)],
        "open": openp, "high": high, "low": low, "close": close,
        "vol": rng.uniform(1e6, 2e6, n),
    })


def test_features_present_and_finite():
    f = candle_features(_ohlcv())
    assert list(f.columns) == FEATURE_COLS
    assert np.isfinite(f.values).all()


def test_scale_invariance():
    df = _ohlcv()
    df2 = df.copy()
    for col in ("open", "high", "low", "close"):
        df2[col] = df2[col] * 10.0                 # vol unchanged
    f1 = candle_features(df).iloc[15:].reset_index(drop=True)
    f2 = candle_features(df2).iloc[15:].reset_index(drop=True)
    assert np.allclose(f1[FEATURE_COLS].values, f2[FEATURE_COLS].values, atol=1e-6)


def test_breakout_flag():
    # 12 flat bars then a clear breakout bar
    n = 13
    df = pd.DataFrame({
        "trade_date": [f"2025{i:04d}" for i in range(n)],
        "open": [10.0] * 12 + [10.0],
        "high": [10.1] * 12 + [12.0],
        "low": [9.9] * 12 + [9.95],
        "close": [10.0] * 12 + [11.9],            # last close >> prior highs
        "vol": [1e6] * n,
    })
    f = candle_features(df, prior=9)
    assert f["breakout"].iloc[-1] == 1.0
    assert f["breakout"].iloc[5] == 0.0           # mid flat region: no breakout
