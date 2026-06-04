# Candlestick-onset alpha1 verdict

## Pooled market-neutral RankIC / net long-short Sharpe

| model | RankIC | net Sharpe |
|---|---|---|
| candle_flat | +0.039 | +0.54 |
| candle_seq | +0.062 | +0.63 |
| factors | +0.061 | +0.81 |
| factors_plus_candle | +0.083 | +1.76 |

Incremental net Sharpe from adding candle geometry: **+0.95**

## SIGN-K1 verdicts

| model | verdict |
|---|---|
| candle_flat_lgbm (K3) | null / not cost-surviving |
| candle_seq (K4) | null / not cost-surviving |
| factors_plus_candle (K5) | REAL (alpha1): pooled net CI excludes 0 AND net>0 in >=2/3 splits |
| factors_only (K5) | null / not cost-surviving |
