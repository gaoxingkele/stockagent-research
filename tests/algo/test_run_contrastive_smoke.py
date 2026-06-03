"""NB5 — hermetic smoke for the contrastive train/eval machinery. CPU, synthetic."""
import numpy as np

from src.identify.run_contrastive import fit_eval


def test_fit_eval_runs():
    rng = np.random.default_rng(0)
    B, T, F = 200, 6, 4
    stock_tr = rng.normal(size=(B, T, F)).astype("float32")
    ref_tr = rng.normal(size=(B, T, F)).astype("float32")
    y_tr = (stock_tr[:, -1, 0] - ref_tr[:, -1, 0]).astype("float32")
    nte = 120
    stock_te = rng.normal(size=(nte, T, F)).astype("float32")
    ref_te = rng.normal(size=(nte, T, F)).astype("float32")
    neutral_te = (stock_te[:, -1, 0] - ref_te[:, -1, 0]).astype("float32")
    raw_te = neutral_te + 0.5 * rng.normal(size=nte).astype("float32")
    dates_te = np.repeat(np.arange(nte // 10), 10).astype(str)

    res = fit_eval(stock_tr, ref_tr, y_tr, stock_te, ref_te, neutral_te, raw_te,
                   dates_te, device="cpu", steps=20, batch=64)
    assert {"market_neutral_rank_ic", "long_short", "final_train_mse"} <= set(res.keys())
    assert {"mean", "lo", "hi"} <= set(res["market_neutral_rank_ic"].keys())
