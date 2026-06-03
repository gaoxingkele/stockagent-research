"""K2 -- dynamic 3-12 bar assembler. CPU, hermetic."""
import numpy as np
import pandas as pd

from src.onset.candle_geometry import FEATURE_COLS
from src.onset.candle_pattern import anchor_features, anchor_sequences


def _panel(seed=0, n=30):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    openp = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({
        "ts_code": ["AAA"] * n,
        "trade_date": [f"2025{i:04d}" for i in range(n)],
        "open": openp, "high": high, "low": low, "close": close,
        "vol": rng.uniform(1e6, 2e6, n),
    })


def test_shapes():
    p = _panel()
    keys = pd.DataFrame({"ts_code": ["AAA"], "trade_date": ["20250025"]})
    flat, mask = anchor_features(p, keys, recent=3, prior=9)
    assert mask[0]
    assert flat.shape == (1, 3 * len(FEATURE_COLS))
    X, m = anchor_sequences(p, keys, window=12, prior=9)
    assert X.shape == (1, 12, len(FEATURE_COLS))


def test_point_in_time_future_bars_do_not_leak():
    p = _panel()
    keys = pd.DataFrame({"ts_code": ["AAA"], "trade_date": ["20250020"]})
    flat_before, _ = anchor_features(p, keys, recent=3, prior=9)
    # corrupt bars AFTER the anchor date with extreme values
    p2 = p.copy()
    fut = p2["trade_date"] > "20250020"
    p2.loc[fut, ["open", "high", "low", "close"]] *= 5.0
    flat_after, _ = anchor_features(p2, keys, recent=3, prior=9)
    assert np.allclose(flat_before.values, flat_after.values, atol=1e-9)
