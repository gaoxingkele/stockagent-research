# Production-edge verdict (does the documented V12.31 mechanism reproduce?)

**VERDICT: NOT REPRODUCIBLE: the documented V12.31 rules (timing + selection) do not reproduce the production edge**

## Decomposition (honest, robust)

| component | result |
|---|---|
| timing (trend regime) incremental Sharpe | +0.08 (net mean CI spans 0 -> wash) |
| disaster_filter fire rate | 0.005 (broken: barely fires, missed 2022 bear) |
| selection pool pooled Sharpe | -1.83 (0/3 positive years) |
| walk-forward-selected OOS Sharpe | [-2.1446902338944134, -2.6603664242791116] (no config recovers an edge) |

Conclusion: the documented rules (onset + V7c filters + simplified disaster timing) do NOT reproduce the production Sharpe 2.20. The real edge lives in parts NOT captured here: the FULL disaster composite (concept signals C2/C3 unimplemented), the actual r20_pred predictive model (we used a momentum proxy), execution/risk/discretion, and production parameter calibration.
