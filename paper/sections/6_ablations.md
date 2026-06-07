# 6. Ablations and Sensitivity

We report four ablation studies: boost-magnitude sensitivity, LLM-model-size sensitivity, expert-prompt sensitivity, and sample-size sensitivity. All ablations use the walk-forward random-sample anchors unless otherwise noted.

## 6.1 Boost Magnitude (E_lgbm_floor_llm_boost)

The E-family router adds a fixed rank-boost $b$ to LGBM-ranked candidates whenever their LLM rank lies in the top quartile. Table 4 reports pooled walk-forward metrics for $b \in \{0.05, 0.15, 0.30, 0.50, 1.00\}$.

**Table 4: Boost Magnitude Ablation (Pooled n=6000)**

| Boost $b$ | RankIC | Top-10% ret | Top-20% ret |
|---|---|---|---|
| 0.00 (= BL_LGBM) | +0.064 | +1.56% | +1.54% |
| 0.15 | +0.075 | +1.45% | +1.43% |
| 0.30 | +0.061 | +1.32% | +1.16% |
| 0.50 | +0.042 | +0.78% | +1.18% |
| 1.00 | (LLM rank only) | +0.32% | +0.36% |

Performance peaks near $b \approx 0.15$ (matching LGBM RankIC at +0.075, with slightly lower Top-10% return). Increasing the boost beyond $b = 0.30$ progressively degrades Top-10% return as LLM-ranked but LGBM-low candidates begin dominating. We interpret this as: the LLM signal carries some marginal information about top-quartile candidates, but treating it as a hard prior (high boost) destroys more LGBM signal than it adds.

## 6.2 LLM Model Size (Sonnet vs Haiku)

We re-run the smoke-test PoC (n=100 stratified) with `claude-haiku-4-5-20251001` to evaluate whether a smaller (and ~30× cheaper) model preserves the expert-prompt effect.

**Table 5: Model Size Comparison (Stratified PoC, n=100)**

| Model + Condition | Top-10% ret | Top-10% WR | Cost |
|---|---|---|---|
| Sonnet 4.6, raw | +1.25% | 50% | $2.27 |
| Sonnet 4.6, expert | **+2.88%** | **60%** | (same) |
| Haiku 4.5, raw | −1.56% | 40% | $0.07 |
| Haiku 4.5, expert | +0.22% | 40% | (same) |

The expert effect ($\Delta_{\text{expert} - \text{raw}}$) is preserved in direction but vastly weaker in magnitude on Haiku (+1.78pp on Haiku vs +1.63pp on Sonnet — comparable!), while the *absolute* signal quality is much lower for Haiku. This decouples the **expert-prompt effect** (relatively model-size-invariant) from the **base model capability** (strongly model-size-dependent). For deployment, the result favors Sonnet over Haiku despite the 30× cost difference, because the absolute Top-10% return determines portfolio P&L while the expert-prompt effect determines marginal improvement only.

## 6.3 Expert Prompt: Stratified vs Random Sample

We isolate the expert-prompt effect $\Delta = \text{LLM}_{\text{expert}} - \text{LLM}_{\text{raw}}$ across sampling regimes:

| Regime | Sample n | Δ Top-10% return | Δ RankIC |
|---|---|---|---|
| Stratified PoC | 1000 | **+0.25pp** | −0.030 |
| Walk-Forward (Split 1) | 2000 | +0.50pp | +0.014 |
| Walk-Forward (Split 2) | 2000 | −0.63pp | −0.048 |
| Walk-Forward (Split 3) | 2000 | −0.23pp | −0.056 |
| WF Pooled | 6000 | **−0.06pp** | −0.030 |

The expert prompt helps marginally in stratified PoC and Split 1, but hurts on Splits 2 and 3 and aggregates to roughly neutral on the pooled walk-forward sample. We hypothesize that the expert prompt overcommits the LLM to the V12.31 onset rules, which work well in onset-rich regimes (PoC stratified, Split 1) but fail when the underlying market regime differs from the deployment context the rules were elicited under (Splits 2–3, post-rotation). This points to **regime-conditioned expert prompt selection** as a future direction.

## 6.4 Sample Size

We evaluate the BL_LGBM and BL_LLM_raw signals on subsets of n ∈ {100, 250, 500, 1000, 2000, 6000} drawn from the walk-forward pool, with 100 bootstrap-replicates per size. Top-10% return for LGBM stabilizes by n=2000 with a 95% CI half-width of approximately ±0.4pp; LLM raw signal stabilizes by n=4000 with half-width ±0.6pp. We use this finding to support our reporting standard of full 6000-anchor pooled bootstrap CIs.

## 6.5 TCN Scaling Law (Pathway 2 Pattern Core)

The Phase 2 TCN+Cross-Attention Pattern Core (Sec. 4.3) exhibits strong scaling behavior as training data increases. On Split 1, holding the architecture fixed (4 dilated conv layers, $d = 64$, 4-head cross-attention, 119,171 parameters), we vary $N_{\text{train}}$:

| $N_{\text{train}}$ | Training time (CPU) | RankIC | Top-10% return | Top-10% WR |
|---|---|---|---|---|
| 20,000 | 46s | +0.012 | +1.88% | 60.0% |
| 100,000 | 208s | +0.084 | +2.48% | 66.3% |
| (LGBM, 3.5M) | (separate) | +0.177 | +3.34% | 69.0% |

A 5× increase in training data yields a 7× improvement in RankIC and +32% improvement in Top-10% return. Extrapolating along this trend, GPU-trained TCN on the full 3.5M-anchor panel would likely match or exceed LGBM performance — closing the remaining gap by an order of magnitude through scale alone. We are CPU-constrained in this work and cannot verify the extrapolation directly; however, Pathway 3 (Section 4.3 and Section 5.8) provides an alternative route by leveraging SSL pretraining on unlabeled anchors to effectively increase the supervised effective sample size.

## 6.6 Do Recent SOTA Regime / Uncertainty Baselines Survive Our Protocol?

A natural objection to our honest-findings (C3, C5, C6) is that a stronger regime detector, an abstention mechanism, or a richer information-theoretic framing would recover a deployable edge where our signals do not. We test this directly by reimplementing the three closest 2025--26 baselines on our data and running them through our exact protocol: leakage-free walk-forward (test windows 2025Q2/Q3/Q4), realistic A-share round-trip cost ($\approx$0.2\%), non-overlapping 5-day periods, date-clustered bootstrap CIs, and the Deflated Sharpe Ratio (DSR) to correct for the number of strategy variants tried [bailey2014deflated]. We do **not** attempt to reproduce their published US-market numbers (different data, task, and mostly no released code); the fair and decisive test is whether their *methods* survive *our* deployment-realistic evaluation. We report this as measurement; that none survive is the point.

**EXP-A --- Regime-Aware LightGBM (rolling-HMM) [regimeawarelgbm2026].** We share one LightGBM cross-sectional ranker across three gates that differ only in the regime condition: no gate (plain), a favorable-state gate from a point-in-time rolling Gaussian-HMM (our reimplementation of the MDPI rolling-HMM), and our own trend-up gate. Pooled net-of-cost results: plain Sharpe $+0.89$ (DSR $0.65$, mean-excess 95\% CI $[-0.0016, +0.0036]$); HMM-gated $+1.92$ (DSR $0.99$, CI $[0.000, +0.0025]$); trend-gated $+1.45$ (DSR $0.82$, CI $[-0.0008, +0.0038]$). The HMM arm is the only one that clears the DSR, but its mean-excess CI lower bound is exactly zero: it has no robustly positive mean edge. Its high Sharpe is a *cash-during-volatility* artifact --- gating to cash in unfavorable regimes shrinks the return volatility (the Sharpe denominator) without creating positive mean excess. This is timing/risk reduction, not selection alpha. No arm has a strictly positive net-of-cost mean CI.

**EXP-B --- Two-level-uncertainty abstention [whenalphabreaks2026].** We attach the When-Alpha-Breaks idea --- abstain (hold cash) when either model-prediction uncertainty (low cross-sectional score dispersion) or regime instability (trailing trend-switch rate) exceeds an expanding past quantile --- to our onset ranker. Pooled net-of-cost: always-on Sharpe $-2.04$ (CI $[-0.0086, -0.0031]$); trend-gate $-1.39$ (CI $[-0.0046, -0.0008]$); abstention $-1.89$ (CI $[-0.0075, -0.0024]$), trading 75\% of periods. Abstention sits between always-on and the trend gate but remains firmly net-negative with a CI strictly below zero. It does not rescue the sub-cost onset edge.

**EXP-C --- Information-theoretic framings [noguer2025fit, mukhia2026conditional].** On the six cross-period-stable hits of Section 5.11, the marginal Normalized Mutual Information is $\le 0.0009$ --- below the $<0.05$ ``efficient-market'' band that Financial Information Theory associates with no exploitable temporal dependence --- so an NMI- or conditional-MI-with-within-stratum-permutation framing would classify these features as carrying essentially no information. Our interaction-information probe (Section 5.11), which permutes the regime *labels*, instead isolates regime-*added* information in 6/6 hits ($p<0.05$). The existing finance information-theory framings would not have surfaced the regime-conditional structure that our probe does.

**Takeaway.** Reimplemented on Chinese A-shares under a leakage-free, cost-aware, cross-period, multiple-testing-corrected protocol, the closest recent regime, abstention, and information-theory baselines are sub-cost --- exactly like our own onset and onset-motif signals. Apparent Sharpe gains from regime gating are cash-during-volatility risk reduction, not net-of-cost alpha. This converts the paper's thesis from ``our signal does not trade'' into the stronger ``even 2025--26 SOTA methods, on this substrate and under honest evaluation, do not trade,'' and it positions the interaction-information probe (C6) as the distinguishing methodological contribution. Figure `bench_summary` plots Sharpe with DSR annotations for all six arms.
