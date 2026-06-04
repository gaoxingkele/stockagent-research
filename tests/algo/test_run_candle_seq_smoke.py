"""K4 -- hermetic smoke for the candle-sequence encoder eval. CPU, synthetic."""
import numpy as np
import pandas as pd

from src.onset.run_candle_seq import fit_eval_seq


def test_fit_eval_seq_runs():
    rng = np.random.default_rng(0)
    B, T, F = 200, 6, 5
    Xtr = rng.normal(size=(B, T, F)).astype("float32")
    ytr = Xtr[:, -1, 0].astype("float32")
    nte = 120
    Xte = rng.normal(size=(nte, T, F)).astype("float32")
    te = pd.DataFrame({"_fwd_r5": rng.normal(size=nte), "mkt_neutral": rng.normal(size=nte),
                       "trade_date": np.repeat(np.arange(nte // 10), 10).astype(str)})
    r = fit_eval_seq(Xtr, ytr, Xte, te, device="cpu", steps=15, batch=64, seeds=(0, 1))
    assert {"rank_ic_market_neutral", "long_short"} <= set(r.keys())
    assert {"mean", "lo", "hi"} <= set(r["rank_ic_market_neutral"].keys())
