# Leakage-free identification of LLM contribution

Leakage validity (A-share) holds: **True**

## Identified estimates (date-clustered 95% CI)

| Estimate | mean | lo | hi | spans 0? |
|---|---|---|---|---|
| ID3 LLM contribution (raw) | +0.033 | -0.037 | +0.110 | yes |
| ID3 LLM contribution (expert) | +0.006 | -0.062 | +0.078 | yes |
| WS2 LLM-weak improvement | -0.149 | -0.298 | -0.009 | no |

## FinBen de-biased (reasoning-only) accuracy

| Benchmark | debiased | memorization excess |
|---|---|---|
| acl | 0.440 | +0.233 |
| bigdata | 0.388 | +0.219 |
| cikm | 0.485 | +0.303 |
