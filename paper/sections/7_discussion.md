# 7. Discussion

## 7.1 Why Does Stratification Inflate LLM Advantage?

A core surprise of this paper is that the apparent +54% Top-10% advantage of hybrid LLM-LGBM routing under stratified sampling reverses to a −15% disadvantage under random sampling that matches the production deployment distribution.

Two mechanisms appear at play.

**Class-imbalance differential sensitivity.** Tree models (LGBM) handle severe class imbalance natively through leaf-level distribution learning, while LLM predictions, trained on uniform distributions across diverse tasks, exhibit miscalibration that compounds in sparse-positive regimes. When the test distribution is artificially balanced (stratification), the LLM's miscalibration is masked; when restored to the true rate, the LGBM's resilient calibration dominates.

**Top-K compression.** At onset rate $\rho = 25\%$ (stratified), the top decile contains predominantly true positives, and any reasonable ranker scores well. At $\rho = 8\%$ (random), the top decile is dominated by near-onset ambiguous anchors, where the LLM's reasoning chain produces qualitatively different (often overconfident) outputs than the LGBM's marginal-probability ranking.

**Implication for the field.** Any future paper claiming LLM-augmented improvement on a movement prediction task should report the sampling protocol, the true onset rate, and at minimum a per-stratum breakdown. We hope this paper triggers an audit of recent finance-LLM benchmarks for similar artifacts.

## 7.2 When Does LLM Add Value? Cross-Quarter Evidence

Even with the disappointing pooled walk-forward results, the per-split breakdown reveals genuine LLM contribution in specific regimes. In Split 2 (2025-Q3), where the LGBM Pattern Core fails (RankIC −0.002), the raw LLM achieves a meaningful Top-10% return of +1.10% — outperforming both LGBM (+0.88%) and all hybrid routers on this split.

We hypothesize three drivers of regime-conditional LLM value:

1. **Distribution shift the LGBM training window did not see.** Split 2's training window ends 2025-03; the test window (2025-Q3) may contain market dynamics structurally different from the training period. LLM agents, trained on broader corpora and reasoning over the per-anchor context, exhibit more graceful degradation.
2. **News-driven onsets.** While our prompts do not include news, the LLM's pretraining corpus contains historical news patterns that may inform calibration under unfamiliar volatility regimes.
3. **Conservative reasoning bias.** Under high uncertainty (Split 2 features anomalous volatility patterns), the LLM's "I don't know" prior softens its probabilities, producing better ranking than overconfident LGBM extrapolation.

Empirically, identifying which quarter falls into which regime *online* is the core challenge for any practical router. We compute the upper bound at +0.39pp Top-10% (oracle) and demonstrate that our online routers capture less than 20% of it; closing this gap is the central topic of Pathway 2.

## 7.3 Pathway 2 Findings and Pathway 3 Outlook

The V12-Agentic framework's modular Pattern Core slot allowed us to implement and evaluate two further phases beyond the LGBM baseline.

**Pathway 2 walk-forward results.** Our Phase 2 TCN+Cross-Attention Pattern Core (Sec. 4.3) achieves equivalent mean Top-10% return to the LGBM baseline (+1.61% vs +1.63%) while training on only 100K anchors per split (vs 3.5M for LGBM) — a 35× data-efficiency advantage. Two specific quarters illuminate the architecture × regime interaction:
- **Split 2 (LGBM failure quarter)**: TCN achieves RankIC +0.034 vs LGBM −0.015, providing meaningful signal where the tree-based model fails. This indicates the TCN encoder captures temporal patterns that gradient boosting on per-anchor features cannot.
- **Split 3 (mixed quarter)**: TCN dominates Top-10% return (+1.54% vs LGBM +0.68%), suggesting better generalization to the most recent test window — a property typically associated with deep models trained on sufficiently large data.

The Phase 2 architecture moreover scales aggressively: Section 6.5 documents a 7× RankIC improvement when training data scales 5× (RankIC +0.012 at 20K → +0.084 at 100K samples). Extrapolating, a GPU-trained TCN on 3.5M anchors would likely match or exceed LGBM — a viable direction for full deployment.

**Pathway 3 (Barlow Twins SSL): the data-efficient bet.** Even with strong scaling, fully training a TCN-based Pattern Core on millions of supervised anchors per split is expensive. The Pathway 3 hypothesis is that pretraining the encoder via Barlow Twins SSL on the full unlabeled D1 panel ($\approx$5M anchors prior to each split's training boundary) and then finetuning on the labeled task (20K–100K anchors) achieves Pathway 2-quality results at a fraction of the supervised data cost. Section 5.8 (when completed) reports Pathway 3 results across the three walk-forward splits.

The Barlow Twins objective is especially well-suited to finance because: (i) it avoids the negative-sample-construction problem (cross-stock cointegration makes "negative" pairs in finance ill-defined); (ii) it scales well to high-dimensional projectors (256-D in our implementation), encouraging fine-grained orthogonal alpha-factor representations; (iii) the augmentations (Gaussian noise, feature dropout, temporal cyclic warp) inject invariances that align with the implicit symmetries of real-market anchor windows.

## 7.4 Limitations

We acknowledge five limitations.

1. **Single-market scope.** The walk-forward evaluation covers only Chinese A-shares. Cross-market validation (NASDAQ, KOSPI, NIFTY) is planned but not in this paper.
2. **3 splits is statistically thin.** We mitigate via 1000-resample bootstrap CIs, but encourage a 6–12 split replication.
3. **Single LLM family.** All LLM experiments use Anthropic Claude (Sonnet 4.6, Haiku 4.5, Opus 4.8). Cross-model replication with GPT-5 / Gemini / Qwen is left for follow-up.
4. **No closed-loop reflection.** The Backtest Verifier reports performance but does not currently trigger upstream agent re-elicitation. Implementing this reflection loop is part of Pathway 2.
5. **Expert prompt is China-specific.** The V12.31 expert prompt encodes A-share-specific rules (ST status, 涨跌停 limits, 龙虎榜). Adaptation to other markets requires re-elicitation; we do not claim direct transferability.
