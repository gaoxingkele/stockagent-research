# Onset-motif: information-theoretic go/no-go

**Verdict:** BUILD-THE-MOTIF-MODEL -- conditional information is permutation-significant AND cross-period stable. The trend/vol regime is a genuine 'transcription factor' that activates the candle 'promoter'.

- MI2 interaction hits (regime adds info): 52/78
- MI3 cross-period-stable items: 6/6
- pooled sample: 200000 rows; 300 permutations; 8 bins

## Top stable interactions (per-year II, nats)

- **close_pct_prior | trend** (stable=True, 4/4 yrs): 2022 +0.0060, 2023 +0.0023, 2024 +0.0103, 2025 +0.0054
- **dist_low_atr | trend** (stable=True, 4/4 yrs): 2022 +0.0059, 2023 +0.0021, 2024 +0.0091, 2025 +0.0056
- **onset_score | trend** (stable=True, 4/4 yrs): 2022 +0.0042, 2023 +0.0022, 2024 +0.0074, 2025 +0.0033
- **body | trend** (stable=True, 4/4 yrs): 2022 +0.0029, 2023 +0.0015, 2024 +0.0034, 2025 +0.0020
- **close_loc | trend** (stable=True, 4/4 yrs): 2022 +0.0030, 2023 +0.0019, 2024 +0.0034, 2025 +0.0012
- **close_pct_prior | vol** (stable=True, 4/4 yrs): 2022 +0.0050, 2023 +0.0044, 2024 +0.0044, 2025 +0.0040

## Honesty caveats (information != alpha)

1. p-values floor at 1/n_perm because n is large (~1.5-2e5): they prove the interaction is >0, NOT that it is large. The effect SIZE (cond_mi ~0.006 nats, correlation-equivalent ~0.07-0.14) is the honest magnitude.
2. INFORMATION != net-of-cost RETURN. The deployability and production-edge lines already showed the long-only, cost-aware return edge collapses cross-period. Conditional information is necessary, not sufficient, for tradable alpha.
3. Mutual information is SIGN-BLIND: part of this conditional information may describe downside/continuation risk in down-trends rather than exploitable upside.
4. The motif model's hard, unsolved job is to convert this stable conditional information into a MONOTONE, costable, long-only signal -- which prior lines suggest is the binding constraint.