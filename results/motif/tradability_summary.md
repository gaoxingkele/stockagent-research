# Onset-motif: tradability verdict (information -> deployment)

**Verdict:** INFORMATION-ONLY (eaten-by-cost) -- the conditional information is directional and monotone, and the regime adds correctly-signed value, but the gross edge is below the transaction-cost floor so net-of-cost return is not positive. Do NOT build a trading motif model; the signal may still serve as a risk/timing overlay or a scientific/methodological contribution.

- build trading motif model: **False** (gross edge < cost floor; gating does help (regime real))
- TRD1 directional+monotone core: True
- TRD2 net-of-cost cross-period survivors: 0/3
- gating helps (regime adds correctly-signed value): True

## The honest chain

1. **MI (onset-motif line):** the trend regime adds STABLE conditional information about candle features -> forward return (significant every year 2022-2025).
2. **TRD1:** that information has a genuine MONOTONE DIRECTIONAL core (conditional-mean rank-corr ~0.09, mono_coef ~1.0) but ~60% of the dispersion is sign-blind variance/risk.
3. **TRD2:** regime-gating IMPROVES Sharpe over ungated (the regime is real), but the gross directional edge (~0.1%/5d) sits BELOW the ~0.2% A-share round-trip cost floor, so net-of-cost Sharpe is negative every year.

**Conclusion:** the onset-motif is REAL, cross-period-stable information with a real directional component -- a defensible scientific/methodological finding -- but it is NOT net-of-cost long-tradable. The binding constraint is transaction cost, consistent with the deployability and production-edge lines. Doing this cheap diagnostic BEFORE building a graph/point-process model avoided building a complex model on an untradable edge.