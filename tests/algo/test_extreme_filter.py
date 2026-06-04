"""FILT1 -- extreme-filter pool. CPU, hermetic."""
import numpy as np
import pandas as pd

from src.onset.extreme_filter import overheated_mask, zombie_mask, pool_mask


def _panel():
    rows = []
    # MOVER: trending stock; ZOMBIE: flat; HOT: ends with a +20% spike
    for code, kind in (("MOVER", "m"), ("ZOMBIE", "z"), ("HOT", "h")):
        base = 10.0
        for i in range(80):
            if kind == "z":
                c = 10.0 + 0.001 * np.sin(i)          # essentially flat
            elif kind == "m":
                c = base * (1.02 ** (i / 10))
            else:
                c = base * (1.01 ** (i / 10)) * (1.2 if i >= 78 else 1.0)  # spike at the end
            rows.append({"ts_code": code, "trade_date": f"2025{i:04d}",
                         "close": c, "low": c * 0.99, "vol": 1e6,
                         "industry": "A", "score": 1.0})
    return pd.DataFrame(rows)


def test_overheated_flags_the_spike():
    p = _panel()
    oh = overheated_mask(p, window=5, thresh=0.08)
    hot_last = p[(p["ts_code"] == "HOT")].index[-1]
    assert oh.loc[hot_last]                                 # +20% spike -> overheated


def test_zombie_flags_the_flat_stock():
    p = _panel()
    z = zombie_mask(p, window=60, flat_thresh=0.03)
    assert z[p["ts_code"] == "ZOMBIE"].iloc[-1]             # flat -> zombie
    assert not z[p["ts_code"] == "MOVER"].iloc[-1]          # trending -> not zombie


def test_pool_is_subset():
    p = _panel()
    m = pool_mask(p, score_col="score", top_pct=0.5)
    assert m.sum() <= len(p)
    assert m.dtype == bool
