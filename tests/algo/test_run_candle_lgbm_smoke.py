"""K3 -- hermetic smoke for candle-LGBM eval machinery."""
import numpy as np
import pandas as pd

from src.onset.run_candle_lgbm import fit_eval_candle, long_short


def test_long_short_gross_net_keys():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"sig": rng.normal(size=n), "_fwd_r5": rng.normal(size=n),
                       "trade_date": np.repeat(np.arange(n // 20), 20).astype(str)})
    ls = long_short(df)
    assert {"gross_sharpe", "net_sharpe", "net_mean_ci95", "n_dates"} <= set(ls.keys())
    assert ls["net_mean"] < ls["gross_mean"]            # cost lowers net


def test_fit_eval_candle_runs():
    rng = np.random.default_rng(1)
    F = 12
    Xtr = rng.normal(size=(400, F)); ytr = Xtr[:, 0] + 0.3 * rng.normal(size=400)
    nte = 200
    Xte = rng.normal(size=(nte, F))
    te = pd.DataFrame({"_fwd_r5": rng.normal(size=nte),
                       "mkt_neutral": rng.normal(size=nte),
                       "trade_date": np.repeat(np.arange(nte // 20), 20).astype(str)})
    r = fit_eval_candle(Xtr, ytr, Xte, te)
    assert "rank_ic_market_neutral" in r and "long_short" in r
    assert {"mean", "lo", "hi"} <= set(r["rank_ic_market_neutral"].keys())
