# Market-neutral identification: beta-timing vs alpha-selection

Leakage validity holds: **True**

## Identified LLM contribution (market-neutral, clustered 95% CI)

| component | signal | mean | lo | hi | spans 0? |
|---|---|---|---|---|---|
| selection (alpha) | raw | -0.004 | -0.064 | +0.054 | yes |
| selection (alpha) | expert | -0.031 | -0.089 | +0.023 | yes |
| timing (beta) | raw | +0.084 | -0.055 | +0.221 | yes |
| timing (beta) | expert | +0.027 | -0.111 | +0.165 | yes |

## Tradable market-neutral long-short (annualized Sharpe; single held-out window -- not alpha evidence)

| source | Sharpe | mean/period | n_dates |
|---|---|---|---|
| NB3 baseline (full window) | 0.59 | +0.0082 | 100 |
| NB5 raw | 2.18 | +0.0095 | 60 |
| NB5 neutral | 1.71 | +0.0070 | 60 |
