# Benchmark synthesis: do 2025-26 SOTA baselines survive our protocol?

**Overall:** NONE of the reimplemented 2025-26 SOTA baselines survive our honest protocol; like our own signals they are sub-cost. The contribution is the honest, deployment-realistic evaluation + the interaction-information probe.

- **EXP-A** (Regime-Aware LightGBM, MDPI 2026): Regime-Aware LightGBM (incl. rolling-HMM) is SUB-COST under our protocol
  - pooled: plain Sharpe=+0.89 DSR=0.65 CI=[-0.0016, 0.0036]; hmm Sharpe=+1.92 DSR=0.99 CI=[0.0, 0.0025]; trend Sharpe=+1.45 DSR=0.82 CI=[-0.0008, 0.0038]
- **EXP-B** (When Alpha Breaks abstention, 2603.13252): two-level abstention does NOT rescue (onset stays sub-cost)
  - pooled: always Sharpe=-2.04 DSR=0.00; trend Sharpe=-1.39 DSR=0.01; abstain Sharpe=-1.89 DSR=0.00
- **EXP-C** (NMI / Conditional-MI info-theory framings): NMI/conditional-MI baselines see an efficient-market band; our interaction info isolates regime-added information (6 hits)

## Takeaway
Reimplemented on A-shares under leakage-free + cost-aware + cross-period + Deflated-Sharpe evaluation, the closest 2025-26 regime/uncertainty baselines are sub-cost just like our own onset/motif signals. Apparent Sharpe gains from regime gating are cash-during-volatility risk-reduction, not net-of-cost alpha. This strengthens the paper's honest-findings thesis and the interaction-information probe (C6) as the distinguishing methodological contribution.