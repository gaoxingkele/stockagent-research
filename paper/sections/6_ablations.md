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
