# 1. Introduction

## 1.1 Motivation: The Industrial-Academic Gap

Stock movement prediction has been studied for decades, yet a persistent gap separates academic methods from deployed industrial systems. Academic work typically evaluates on synthetic benchmarks or stratified subsets that overrepresent the events of interest (e.g., 25% positive class in many "movement onset" datasets), while deployed systems must operate under the true marginal distribution (typically <10% positive rate). This sampling gap is consequential: methods that appear to outperform baselines on stratified data may degrade — or even underperform — once exposed to realistic class distributions.

Recent advances in large language model (LLM) agents have intensified this gap. LLM-based financial assistants (FinAgent (Yu et al. 2024), AlphaForge (Hu et al. 2024), TradingGPT (Li et al. 2024)) demonstrate impressive reasoning capabilities on curated tasks. Yet whether these agents add genuine alpha when integrated into a production pipeline — and how to combine their predictions with mature statistical models such as gradient-boosted trees — remains poorly studied at scale.

We address this gap through **V12-Agentic**, a 4-agent framework grounded in a deployed quantitative trading system (V12.31, real α +2.236pp/month, Sharpe 2.20) and rigorously validated through walk-forward evaluation on a real-distribution sample of 6,000 stock-day anchors drawn from the Chinese A-share market.

## 1.2 The A-share Setting and Asymmetric Constraints

The Chinese A-share market is an ideal testbed for several reasons:

1. **Short-sale constraint**: Most retail-accessible A-shares cannot be shorted, making bearish signals an avoidance criterion rather than a tradeable direction. This induces fundamental asymmetry in label design and agent decision rules.
2. **Sparse onset events**: Under the production V12.31 v3c three-class label (future 5-day return ≥ 10% with drawdown ≤ 5%, conditioned on past 5-day return ≤ 8%), the bullish onset rate is approximately 8% across our 5.27M anchor universe — closely matching empirical rare-event distributions.
3. **Disaster month dynamics**: Production experience identifies recurring "disaster months" (e.g., 202602, 202603) where conventional momentum strategies suffer 3-5σ drawdowns. Detecting and routing around these regimes is a core operational concern.

These three properties — asymmetric direction, sparse events, regime non-stationarity — collectively define the **movement onset detection** problem we address.

## 1.3 V12-Agentic: Four Agents for Onset Detection

Inspired by how the V12.31 production system is actually operated, we decompose the onset detection task into four cooperating agents:

| Agent | Function |
|---|---|
| **Macro Regime Monitor** | Aggregates index returns, volume signals, sector breadth, and concept heat into a real-time "disaster month" flag using a vote-2/3 composite rule. |
| **Alpha Factor Explorer** | Maintains 153 technical/fundamental factors and 8 concept heat factors. Encodes deployed expert knowledge (Round 1-3 interview transcripts) as a structured natural-language prompt for downstream LLM agents. |
| **Pattern Core** | The core onset classifier. In this paper (Pathway 1) it is a LightGBM multi-class model trained with forward-label-leakage protection. Future work (Pathway 2-3) replaces this with TCN + Spatio-Temporal Cross-Attention and Barlow-Twins pretrained representations. |
| **Backtest & Verifier** | Combines Pattern Core probabilities with LLM agent predictions through a confidence-routed hybrid signal. Performs walk-forward evaluation and reports bootstrap confidence intervals. |

The architecture is deliberately modular: each agent can be upgraded independently. We demonstrate this by reporting both a "Phase 1" instantiation (LGBM Pattern Core) and an oracle upper bound for any quarter-conditional router.

## 1.4 Contributions

This paper makes five contributions. The core novelty is methodological — the knowledge-encoding procedure (C1), the V12-Agentic framework anchored to a real deployed system (C2), and a leakage-free *identification* framework for the LLM's contribution (C5); C3–C4 are honest, deployment-realistic empirical findings rather than new techniques.

**C1 — Knowledge Encoding Methodology**. We document a systematic procedure for converting tacit production knowledge — accumulated through multi-year deployment, code commits, post-mortems, and operator interviews — into a structured natural-language prompt usable by LLM agents. Applied to V12.31, this procedure produces a ~5,500-character expert prompt covering onset definition, asymmetric direction rules, V7c five iron rules, and disaster-month signals (Sections 4.2 and Appendix A).

**C2 — V12-Agentic Architecture**. We propose a 4-agent decomposition (Macro Regime Monitor + Alpha Factor Explorer + Pattern Core + Backtest Verifier) that formalizes industrial best practices in a paper-grade framework (Section 4). Each agent maps to a concrete implementation drawn from the V12.31 production codebase, ensuring reproducibility and deployment realism.

**C3 — Hybrid-Over-Baseline Effects Are Within Noise**. On a 1,000-anchor stratified sample our best hybrid router appears to beat a pure-LGBM baseline by +54% Top-10% return (+1.93% vs +1.25%); on a 6,000-anchor walk-forward random sample the same router trails by −15%. We show this "reversal" is not a base-rate mechanism but a noise artifact, via three controlled experiments on the already-scored anchors (Section 5.5): (a) no onset-rate or expert-score composition of the random pool reproduces hybrid > LGBM, ruling out base-rate compression; (b) swapping in a stronger LGBM base ranker leaves the hybrid edge unchanged, ruling out baseline weakness; and (c) under a **date-clustered** bootstrap the headline gap is +0.72pp with 95% CI [−0.56pp, +2.05pp], spanning zero. Our contribution here is the honest deployment-realistic *negative result*, not the statistics: cluster-robust inference for cross-sectionally correlated returns is long established (clustered standard errors, Fama-MacBeth), and we simply apply it. The takeaway is that at the sample sizes typical of LLM-finance studies, hybrid-over-baseline effects of this magnitude are statistically indistinguishable from zero — so ratio-reported "LLM helps" claims without cluster-robust intervals are unreliable.

**C4 — Quarter-Conditional Headroom**. Through paired-split analysis (Section 5.7), we identify quarter-level heterogeneity in LGBM-LLM complementarity: in our 3-split walk-forward, LGBM dominates Split 1, raw LLM dominates Split 2, and a hybrid dominates Split 3. We compute an oracle upper bound (+0.34pp Top-10% return relative to LGBM) that bounds the achievable gain from any online quarter-conditional router. Five router variants achieve ~20% of this oracle (a marginal but real signal), suggesting fertile ground for richer regime-conditioning future work.

**C5 — Leakage-free identification of the LLM's contribution.** Because the field's "LLM helps" results are confounded by memorization (it cannot be separated from reasoning on contaminated data), the *contribution of LLM reasoning is unidentified*. We turn the leakage-free A-share substrate into an identification condition — the no-context probe at chance (48.6%) certifies the memory channel is null — and build (i) a leakage-validity-gated estimator of the LLM's incremental contribution over a tabular baseline (partial rank correlation, date-clustered CIs), and (ii) a leakage-calibrated *de-biasing* estimator that uses the clean market to subtract the memorization component from contaminated-benchmark scores (Section 5.10). This yields the first *identified* (not memory-confounded) estimate of LLM value-add in stock-movement onset prediction — which is ≈0 (raw +0.033 [−0.037, +0.110]; expert +0.006 [−0.062, +0.078]); LLM-as-weak-supervisor is in fact harmful (−0.149 [−0.298, −0.009]); and de-biasing collapses FinBen's 67–80% to at-or-below chance (0.39–0.49). Concurrent work separates memory from reasoning via masking on the CSI300 [ktdfin2026] and quantifies agent leakage [profitmirage2025]; our distinct elements are the explicit identification condition and the cross-market de-biasing estimator that exports a correction to existing contaminated benchmarks.

**Why a leakage-resistant 2025 benchmark (not a novel claim, but a prerequisite).** The four contributions above are deliberately evaluated on a 2025 walk-forward sample that post-dates common LLM training cutoffs. This is not optional: a growing body of work shows that LLMs evaluated on pre-cutoff financial data measure *memorization*, not forecasting — look-ahead and "distraction" bias in GPT sentiment trading [glasserman2023lookahead], lookahead bias in pretrained LMs [sarkar2024lookahead], the need for post-cutoff samples [lopezlira2023chatgpt], information-leakage "profit mirages" in LLM agents [profitmirage2025], and identifier/calendar masking on the CSI300 [ktdfin2026]. Consistent with this literature, we confirm (Section 5.9, **not claimed as novel**) that on the FinBen/PIXIU stock-movement suite (ACL18/BigData22/CIKM18) gemini-3.5-flash scores 67–80% directional accuracy but *as high or higher with no input at all* (only the ticker and date), with a mega-cap gradient and a cross-market control (the same protocol on barely-memorized A-shares falls to chance, 48.6%). We include this only to justify why C1–C4 are measured on leakage-resistant data.

## 1.5 Paper Outline

Section 2 reviews related work in stock movement prediction, LLM finance agents, and self-supervised time-series representation. Section 3 formalizes the movement onset problem with asymmetric short-sale constraints. Section 4 presents the V12-Agentic architecture in detail. Section 5 reports experiments across 6,000 walk-forward anchors, 14 baselines and hybrid variants, and three sampling regimes. Section 6 ablates boost magnitude, LLM model size, and expert-prompt sensitivity. Section 7 discusses implications, and Section 8 concludes with concrete next steps for Pathways 2 and 3.

All code, data preprocessing, and evaluation scripts are released as supplementary material upon acceptance. The deployed V12.31 system (production code, real α records, and operator transcripts) is referenced for context but not released in raw form due to commercial considerations; we instead release a faithful reproduction in the form of the natural-language expert prompt (Appendix A).
